# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Construction of the master pipeline.
"""

from typing import Dict
from kedro.pipeline import Pipeline



###########################################################################
# Here you can find an example pipeline, made of two modular pipelines.
#
# Delete this when you start working on your own Kedro project as
# well as pipelines/data_science AND pipelines/data_engineering
# -------------------------------------------------------------------------

from pokemon_challenge.pipelines import preprocessing
from pokemon_challenge.pipelines import optimization
from pokemon_challenge.pipelines import modelling


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """

    preprocessing_pipeline = preprocessing.preprocessing_pipeline()
    lp_optimization_pipeline = optimization.lp_optimization_pipeline()
    ga_optimization_pipeline = optimization.ga_optimization_pipeline()
    fit_dnn_pipeline = modelling.fit_dnn_pipeline()
    eval_dnn_pipeline = modelling.eval_dnn_pipeline()
    prediction_pipeline = modelling.prediction_pipeline()

    return {
        "preprocessing_pipeline": preprocessing_pipeline,
        "lp_optimization": lp_optimization_pipeline,
        "ga_optimization": ga_optimization_pipeline,
        "dnn_pipeline": fit_dnn_pipeline,
        "eval_dnn_pipeline": eval_dnn_pipeline,
        "prediction_pipeline": prediction_pipeline,
        "__default__": preprocessing_pipeline + prediction_pipeline + lp_optimization_pipeline,
    }

