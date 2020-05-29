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

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""

from kedro.pipeline import Pipeline, node

from .nodes import fit_dnn_model
from .nodes import eval_dnn_models
from .nodes import predict_available_pokemon_performance


def fit_dnn_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                fit_dnn_model,
                dict(
                    X_train="x_train",
                    y_train="y_train",
                    X_val="x_val",
                    y_val="y_val",
                    X_test="x_test",
                    y_test="y_test",
                    dnn_params="params:dnn_params",
                    dnn_arch="params:dnn_test"
                ),
                None,
            ),
        ]
    )

def eval_dnn_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                eval_dnn_models,
                dict(
                    X_train="x_train",
                    y_train="y_train",
                    X_val="x_val",
                    y_val="y_val",
                    X_test="x_test",
                    y_test="y_test",
                    dnn_arch_1="params:dnn_m1",
                    dnn_arch_2="params:dnn_m2",
                    dnn_arch_3="params:dnn_m3",
                    dnn_arch_4="params:dnn_m4",
                ),
                None,
            ),
        ]
    )

def prediction_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                predict_available_pokemon_performance,
                dict(
                    available_pokemon="available_pokemon",
                    battles="available_battles_preprocessed",
                    dnn_arch_1="params:dnn_m1",
                    dnn_arch_2="params:dnn_m2",
                    dnn_arch_3="params:dnn_m3",
                    dnn_arch_4="params:dnn_m4",
                ),
                ["available_pokemon_performance","all_predictions"],
            ),
        ]
    )
