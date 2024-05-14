from tsg.parameters_generation.aggregation_method import AggregationMethod
from tsg.parameters_generation.parametrization_method import ParametrizationMethod
from tsg.parameters_generation.random_method import RandomMethod

ALL_GENERATION_METHODS = {
    RandomMethod.name: RandomMethod,
    AggregationMethod.name: AggregationMethod,
    ParametrizationMethod.name: ParametrizationMethod,
}
