from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vlmserve.sampling import SamplingParams

MAX_STOP_STRINGS = 16


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: str | None = None  # accepted and ignored


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = Annotated[TextContent | ImageContent, Field(discriminator="type")]


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[ContentPart] | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    # clients send plenty of fields we have no use for (user, seed, ...): ignore those
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0)  # not openai, but everybody has it
    repetition_penalty: float | None = Field(default=None, gt=0)  # same here
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: int = Field(default=1, ge=1, le=1)

    # Things that change the answer and that we don't do. Better a clear 400 than an
    # answer that quietly leaves them out.
    tools: Any = None
    tool_choice: Any = None
    functions: Any = None
    function_call: Any = None
    response_format: Any = None
    logit_bias: Any = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    @model_validator(mode="after")
    def refuse_what_we_cannot_do(self) -> "ChatCompletionRequest":
        unsupported = [
            name
            for name in (
                "tools",
                "tool_choice",
                "functions",
                "function_call",
                "logit_bias",
            )
            if getattr(self, name)
        ]
        if self.logprobs or self.top_logprobs:
            unsupported.append("logprobs")
        if self.frequency_penalty or self.presence_penalty:
            unsupported.append("frequency_penalty/presence_penalty")
        fmt = self.response_format
        if fmt and not (isinstance(fmt, dict) and fmt.get("type", "text") == "text"):
            unsupported.append("response_format")
        if unsupported:
            raise ValueError(f"not supported by this server: {', '.join(unsupported)}")
        if isinstance(self.stop, list) and len(self.stop) > MAX_STOP_STRINGS:
            raise ValueError(f"at most {MAX_STOP_STRINGS} stop strings")
        return self

    def sampling_params(self, room: int) -> SamplingParams:
        """room is how many tokens of the context window the prompt left free."""
        asked = self.max_completion_tokens or self.max_tokens or room
        stop = [self.stop] if isinstance(self.stop, str) else (self.stop or [])
        return SamplingParams(
            max_tokens=min(asked, room),
            temperature=1.0 if self.temperature is None else self.temperature,
            top_p=1.0 if self.top_p is None else self.top_p,
            top_k=self.top_k or 0,
            repetition_penalty=self.repetition_penalty or 1.0,
            stop=tuple(stop),
        )


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str | None


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "vlmserve"
    max_model_len: int


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class ErrorBody(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
