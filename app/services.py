"""Process-wide singletons: Redis store, graph store, tracer, event bus."""
import threading

from app.observability import EventBus, Tracer
from app.state import graph_store
from app.state.redis_store import RedisStore


class Services:
    def __init__(self):
        self._lock = threading.Lock()
        self._store = self._graph = self._tracer = None
        self.bus = EventBus()

    @property
    def store(self) -> RedisStore:
        with self._lock:
            if self._store is None:
                self._store = RedisStore()
            return self._store

    @property
    def graph(self):
        with self._lock:
            if self._graph is None:
                self._graph = graph_store.connect()
            return self._graph

    @property
    def tracer(self) -> Tracer:
        store = self.store
        with self._lock:
            if self._tracer is None:
                self._tracer = Tracer(store, self.bus)
            return self._tracer


services = Services()
