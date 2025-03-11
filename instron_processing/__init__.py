import pint
import pint_pandas
from pint import UnitRegistry, set_application_registry

ureg = UnitRegistry()
ureg.formatter.default_format = "P"
set_application_registry(ureg)
Q_ = ureg.Quantity
