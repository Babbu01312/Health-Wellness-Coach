from deepeval.metrics import HallucinationMetric,ContextualRelevancyMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.llms.ollama_model import OllamaModel  # Import the base OllamaModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class HallucinationEvaluator:
    def __init__(self, context: list, prompt: str, actual_output: str, model_name="mistral", threshold=0.5):
        """
        Initialize the evaluator with a custom model and threshold.
        """
        self.model = CustomOllamaMistralModel(model_name=model_name)
        self.test_case = LLMTestCase(
            input=prompt,
            actual_output=actual_output,
            retrieval_context=context,
            context=context
        )
        self.threshold = threshold
        

    def hallucination(self) -> dict:

        # Create and run hallucination metric
        hallucination_metric = HallucinationMetric(threshold=self.threshold, model=self.model)
        print(hallucination_metric)
        try:
            hallucination_score = hallucination_metric.measure(self.test_case)
            logging.info("Hallucination metric measured successfully.")
        except Exception as e:
            logging.error(f"Error measuring hallucination metric: {e}")
            hallucination_score = None

        # Return results
        return {"hallucination_score": hallucination_score if hallucination_score is not None else "N/A"}
    def contextual_relavancy(self) -> dict:
        """
        Calculate contextual relavancy using the test case and model.
        """
        try:
            # Create and run contextual relavancy metric
            contextual_relavancy_metric = ContextualRelevancyMetric(model=self.model)
            contextual_relavancy_score = contextual_relavancy_metric.measure(self.test_case)
            logging.info("Contextual relavancy metric measured successfully.")
        except Exception as e:
            logging.error(f"Error measuring contextual relavancy metric: {e}")
            contextual_relavancy_score = None

        # Return results
        return {"contextual_relavancy_score": contextual_relavancy_score if contextual_relavancy_score is not None else "N/A"}
    def answer_relevancy(self) -> dict:
        """
        Calculate answer relevancy using the test case and model.
        """
        try:
            # Create and run answer relevancy metric
            answer_relevancy_metric = AnswerRelevancyMetric(model=self.model)
            answer_relevancy_score = answer_relevancy_metric.measure(self.test_case)
            logging.info("Answer relevancy metric measured successfully.")
        except Exception as e:
            logging.error(f"Error measuring answer relevancy metric: {e}")
            answer_relevancy_score = None

        # Return results
        return {"answer_relevancy_score": answer_relevancy_score if answer_relevancy_score is not None else "N/A"}

class CustomOllamaMistralModel(OllamaModel):
    def __init__(self, model_name="mistral"):
        super().__init__(model=model_name)  # Initialize the base OllamaModel with the model name


# HallucinationEvaluator = HallucinationEvaluator(
#     context=["This is a test context."],
#     prompt="What is the capital of France?",
#     actual_output="Paris",
#     model_name="mistral",
#     threshold=0.5
# )

# print(HallucinationEvaluator.contextual_precision())