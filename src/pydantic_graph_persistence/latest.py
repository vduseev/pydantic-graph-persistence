from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Annotated, Awaitable

import pydantic
from pydantic_graph import exceptions
from pydantic_graph.nodes import (
    BaseNode,
    End,
    StateT,
)
from pydantic_graph.persistence import (
    BaseStatePersistence,
    EndSnapshot,
    NodeSnapshot,
    RunEndT,
    Snapshot,
    SnapshotStatus,
    _utils,
)


@dataclass
class LatestStatePersistence(BaseStatePersistence[StateT, RunEndT]):
    """State persistence that stores the latest snapshot of the graph state.
    
    This persistence is useful for storing the latest state of the graph. You
    must provide a way to load and save the state snapshot.

    Args:
        load: Callback to load the state snapshot from somewhere. Accepts
            no arguments and returns a dict with the latest state snapshot.
        save: Callback to save the state snapshot to somewhere. Accepts a
            dict with the latest state snapshot and returns nothing.
    """

    load: Callable[[], Awaitable[dict]]
    """Callback to load the state snapshot from somewhere.
    
    Returns:
        dict: The latest state snapshot.
    """
    save: Callable[[dict], Awaitable[None]]
    """Callback to save the state snapshot to somewhere.
    
    Args:
        snapshot (dict): The latest state snapshot to save.
    """

    _snapshot_type_adapter: (
        pydantic.TypeAdapter[Snapshot[StateT, RunEndT]] | None
    ) = field(default=None, init=False, repr=False)

    async def snapshot_node(
        self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
    ) -> None:
        # Save current node snapshot in the database
        snapshot = NodeSnapshot(state=state, node=next_node)
        await self._save(snapshot)

    async def snapshot_node_if_new(
        self,
        snapshot_id: str,
        state: StateT,
        next_node: BaseNode[StateT, Any, RunEndT],
    ) -> None:
        last_snapshot = await self._load()
        if last_snapshot and last_snapshot.id == snapshot_id:
            return
        else:
            await self.snapshot_node(state, next_node)

    async def snapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
        snapshot = EndSnapshot(state=state, result=end)
        await self._save(snapshot)

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        last_snapshot = await self._load()
        if last_snapshot is None or snapshot_id != last_snapshot.id:
            raise LookupError(f"No snapshot found with id={snapshot_id!r}")

        assert isinstance(last_snapshot, NodeSnapshot), (
            "Only NodeSnapshot can be recorded"
        )
        exceptions.GraphNodeStatusError.check(last_snapshot.status)
        last_snapshot.status = "running"
        last_snapshot.start_ts = _utils.now_utc()
        await self._save(last_snapshot)

        start = perf_counter()
        try:
            yield
        except Exception:
            duration = perf_counter() - start
            await self._after_run(snapshot_id, duration, "error")
            raise
        else:
            duration = perf_counter() - start
            await self._after_run(snapshot_id, duration, "success")

    async def load_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
        last_snapshot = await self._load()
        if (
            isinstance(last_snapshot, NodeSnapshot)
            and last_snapshot.status == "created"
        ):
            last_snapshot.status = "pending"
            await self._save(last_snapshot)
            return last_snapshot

    async def load_all(self) -> list[Snapshot[StateT, RunEndT]]:
        raise NotImplementedError(
            "load is not supported for DatabaseStatePersistence"
        )

    def should_set_types(self) -> bool:
        """Whether types need to be set."""
        return self._snapshot_type_adapter is None

    def set_types(
        self, state_type: type[StateT], run_end_type: type[RunEndT]
    ) -> None:
        self._snapshot_type_adapter = build_snapshot_type_adapter(
            state_type, run_end_type
        )

    async def _load(self) -> Snapshot[StateT, RunEndT] | None:
        assert self._snapshot_type_adapter is not None, (
            "snapshot type adapter must be set"
        )
        data = await self.load()
        return self._snapshot_type_adapter.validate_python(data)

    async def _save(self, snapshot: Snapshot[StateT, RunEndT]) -> None:
        assert self._snapshot_type_adapter is not None, (
            "snapshot type adapter must be set"
        )
        data = self._snapshot_type_adapter.dump_python(snapshot, mode="json")
        await self.save(data)

    async def _after_run(
        self, snapshot_id: str, duration: float, status: SnapshotStatus
    ) -> None:
        snapshot = await self._load()
        if snapshot is None or snapshot.id != snapshot_id:
            raise LookupError(f"No snapshot found with id={snapshot_id!r}")
        assert isinstance(snapshot, NodeSnapshot), (
            "Only NodeSnapshot can be recorded"
        )
        snapshot.status = status
        snapshot.duration = duration
        await self._save(snapshot)


def build_snapshot_type_adapter(
    state_t: type[StateT], run_end_t: type[RunEndT]
) -> pydantic.TypeAdapter[Snapshot[StateT, RunEndT]]:
    """Build a type adapter for a snapshot.

    This method should be called from within
    [`set_types`][pydantic_graph.persistence.BaseStatePersistence.set_types]
    where context variables will be set such that Pydantic can create a schema
    for [`NodeSnapshot.node`][pydantic_graph.persistence.NodeSnapshot.node].
    """
    return pydantic.TypeAdapter(
        Annotated[Snapshot[state_t, run_end_t], pydantic.Discriminator("kind")]
    )
