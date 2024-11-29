import json
import os
from typing import Any, Dict, List, Optional

try:
    import boto3
except ImportError:
    raise ImportError("The 'boto3' library is required. Please install it using 'pip install boto3'.")

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase


class AWSBedrockLLM(LLMBase):
    def __init__(self, config: Optional[BaseLlmConfig] = None):
        super().__init__(config)

        if not self.config.model:
            self.config.model = "anthropic.claude-3-5-sonnet-20240620-v1:0"

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

        self.client = self.session.client("bedrock-runtime")
        
        self.model_kwargs = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

    def _format_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Formats messages and system prompt for the Bedrock Converse API.

        Args:
            messages (List[Dict[str, str]]): A list of dictionaries where each dictionary represents a message.
                                           Each dictionary contains 'role' and 'content' keys.

        Returns:
            Dict[str, Any]: A dictionary containing formatted messages and system prompt if present.
        """
        formatted_request = {}
        formatted_messages = []
        system_content = []

        for message in messages:
            role = message["role"].lower()
            content = message["content"]

            if role == "system":
                system_content.append({"text": content})
            elif role in ["user", "assistant"]:
                formatted_messages.append({
                    "role": role,
                    "content": [{"text": content}]
                })

        formatted_request["messages"] = formatted_messages
        if system_content:
            formatted_request["system"] = system_content

        return formatted_request

    def _parse_response(self, response, tools) -> str:
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {"tool_calls": []}

            if response["output"]["message"]["content"]:
                for content_block in response["output"]["message"]["content"]:
                    if "toolUse" in content_block:
                        processed_response["tool_calls"].append(
                            {
                                "name": content_block["toolUse"]["name"],
                                "arguments": content_block["toolUse"]["input"],
                            }
                        )

            return processed_response

        # For non-tool responses, extract text from content blocks
        if "output" in response and "message" in response["output"]:
            message = response["output"]["message"]
            if "content" in message:
                text_blocks = [block["text"] for block in message["content"] if "text" in block]
                return " ".join(text_blocks)
        return ""

    def _prepare_inference_config(self, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares the inference configuration for the model.

        Args:
            model_kwargs (Dict[str, Any]): Model-specific keyword arguments.

        Returns:
            Dict[str, Any]: The prepared inference configuration.
        """
        inference_config = {}
        if "max_tokens" in model_kwargs:
            inference_config["maxTokens"] = model_kwargs["max_tokens"]
        if "temperature" in model_kwargs:
            inference_config["temperature"] = model_kwargs["temperature"]
        if "top_p" in model_kwargs:
            inference_config["topP"] = model_kwargs["top_p"]
        return inference_config

    def _convert_tool_format(self, original_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Converts a list of tools from their original format to the format required by the Bedrock Converse API.

        Args:
            original_tools (List[Dict[str, Any]]): A list of tools in the original format.

        Returns:
            List[Dict[str, Any]]: A list of tools in the Bedrock Converse API format.
        """
        new_tools = []

        for tool in original_tools:
            if tool["type"] == "function":
                function = tool["function"]
                new_tool = {
                    "toolSpec": {
                        "name": function["name"],
                        "description": function.get("description", ""),
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {},
                                "required": function["parameters"].get("required", []),
                            }
                        },
                    }
                }

                for prop, details in function["parameters"].get("properties", {}).items():
                    new_tool["toolSpec"]["inputSchema"]["json"]["properties"][prop] = {
                        "type": details.get("type", "string"),
                        "description": details.get("description", ""),
                    }

                new_tools.append(new_tool)

        return new_tools

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using AWS Bedrock.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format: Not used in this implementation.
            tools (list, optional): List of tools that the model can call.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        formatted_request = self._format_messages(messages)
        request_body = {
            "modelId": self.config.model,
            **formatted_request
        }

        if tools:
            tool_config = {
                "tools": self._convert_tool_format(tools),
                "toolChoice": {"auto": {}} if tool_choice == "auto" else {"any": {}}
            }
            request_body["toolConfig"] = tool_config

        inference_config = self._prepare_inference_config(self.model_kwargs)
        if inference_config:
            request_body["inferenceConfig"] = inference_config

        response = self.client.converse(**request_body)
        return self._parse_response(response, tools)
