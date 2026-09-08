"""
Storage — models.py (Day 2 implementation).

SQLAlchemy ORM models for the structured log store.
These models track every node invocation for post-run analysis.

Stub for Day 1 — full implementation in Day 2.
"""

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# TODO (Day 2): Implement the following tables:
#
# class Run(Base):
#     __tablename__ = "runs"
#     id = Column(String, primary_key=True)       # UUID
#     query = Column(Text)
#     status = Column(String)                      # completed / crashed / breaker_triggered
#     created_at = Column(DateTime)
#     iterations = Column(Integer)
#
# class IterationLog(Base):
#     __tablename__ = "iteration_logs"
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     run_id = Column(String, index=True)
#     iteration = Column(Integer)
#     node = Column(String)                        # researcher / critic / writer
#     embedding = Column(Text)                     # JSON-serialized float list
#     prompt_tokens = Column(Integer)
#     completion_tokens = Column(Integer)
#     timestamp = Column(DateTime)
#     breaker_state = Column(String)               # CLOSED / HALF_OPEN / OPEN
