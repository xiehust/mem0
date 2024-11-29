from typing import Optional

import os
try:
    import boto3
except ImportError:
    raise ImportError("The 'boto3' library is required. Please install it using 'pip install boto3'.")
import json
from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase
from typing import Any, Dict, List, Optional

class AWSBedrockEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "amazon.titan-embed-text-v2:0"
        self.config.embedding_dims = self.config.embedding_dims or 1024
        self.model_kwargs = self.config.model_kwargs or {}

        self.aws_access_key=os.environ.get("AWS_ACCESS_KEY")
        self.aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.region = os.environ.get("AWS_REGION")

        if self.aws_access_key and self.aws_secret_key:
            self.session = boto3.Session(
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.region
            )
        else:
            self.session = boto3.Session(region_name=self.region)

        self.client = self.session.client(
            "bedrock-runtime")

    def _embedding_func(self, text: str) -> List[float]:
        """Call out to Bedrock embedding endpoint."""
        # replace newlines, which can negatively affect performance.
        text = text.replace(os.linesep, " ")

        # format input body for provider
        provider =  self.config.model.split(".")[0]
        _model_kwargs = self.model_kwargs
        input_body = {**_model_kwargs}
        if provider == "cohere":
            if "input_type" not in input_body.keys():
                input_body["input_type"] = "search_document"
            input_body["texts"] = [text]
        else:
            # includes common provider == "amazon"
            input_body["inputText"] = text
        body = json.dumps(input_body)

        try:
            response = self.client.invoke_model(
                body=body,
                modelId=self.config.model,
                accept="application/json",
                contentType="application/json",
            )

            # format output based on provider
            response_body = json.loads(response.get("body").read())
            if provider == "cohere":
                return response_body.get("embeddings")[0]
            else:
                # includes common provider == "amazon"
                return response_body.get("embedding")
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")
        
    def embed(self, text):
        """
        Get the embedding for the given text using Vertex AI.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        embeddings = self._embedding_func(text)
        return embeddings
    
