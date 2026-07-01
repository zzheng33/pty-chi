import enum

try:
    StrEnum = enum.StrEnum
except AttributeError:
    class StrEnum(str, enum.Enum):
        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

