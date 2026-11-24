"""Structured doctor results. No terminal rendering lives here."""

from enum import StrEnum

from pydantic import BaseModel


class Status(StrEnum):
    OK = "OK"
    WARNING = "WARNING"
    ERROR = "ERROR"


class CheckResult(BaseModel):
    name: str
    status: Status
    detail: str = ""


class DoctorReport(BaseModel):
    results: list[CheckResult]

    @property
    def has_errors(self) -> bool:
        return any(r.status is Status.ERROR for r in self.results)

    def count(self, status: Status) -> int:
        return sum(1 for r in self.results if r.status is status)
