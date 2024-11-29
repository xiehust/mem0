from typing import Any, Dict

from pydantic import BaseModel, Field, model_validator


class AWSOpenSearchConfig(BaseModel):
    host: str = Field(None, description="OpenSearch domain endpoint (without https://)")
    collection_name: str = Field("mem0", description="Name of the collection")
    use_iam: bool = Field(False, description="Whether to use IAM authentication instead of basic auth")
    embedding_model_dims: int = Field(None, description="Dimension of the embedding vector")
    secret_arn: str = Field(None, description="ARN of the secret in AWS Secrets Manager containing credentials.")
    username: str = Field(None, description="Username for basic authentication")
    password: str = Field(None, description="Password for basic authentication")

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    model_config = {
        "arbitrary_types_allowed": True,
    }
