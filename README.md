# LLMConnector

`llmconnector` is a Go package designed to facilitate communication with various Large Language Model (LLM) APIs, such as OpenAI and Alibaba. It provides a unified interface for chat and embedding functionalities, making it easier to integrate and switch between different LLM providers.

## Features

- **Chat Operations**: Interact with language models to generate text based on input messages.
- **Embedding Operations**: Generate embeddings for text inputs, useful for various NLP tasks.
- **Strategy Pattern**: Supports multiple strategies for chat and embedding operations, allowing easy integration of new providers.
- **Customizable Options**: Provides a variety of options to customize chat and embedding requests, such as model type, temperature, max tokens, and more.
- **Advanced HTTP Client**: Uses `github.com/simp-lee/gohttpclient` for robust HTTP operations with features like retries, rate limiting, and proxy support.
- **Extensible**: Easy to add support for new language model providers.
- **Concurrent-Safe**: Designed to be used in concurrent environments.
- **Common Configuration**: Shared configuration options across different providers for consistency and ease of use.

## Installation

To install LLMConnector, use the following command:

```shell
go get github.com/simp-lee/llmconnector
```

## Usage

### Setting Up the Model Context

```go
package main

import (
	"context"
	"fmt"
	"github.com/simp-lee/llmconnector"
)

func main() {
	ctx := context.Background()

	// Create a new model context
	modelContext := llmconnector.NewModelContext()

	// Set up OpenAI strategy for chat
	openAIConfig := llmconnector.Config{
		APIKey: "your-openai-api-key",
	}
	openAIStrategy, err := llmconnector.NewOpenAIStrategy(openAIConfig)
	if err != nil {
		fmt.Println("Error setting up OpenAI strategy:", err)
		return
	}
	modelContext.SetChatStrategy(openAIStrategy)
	
	// Set up Alibaba strategy for embedding
	alibabaConfig := llmconnector.Config{
		APIKey: "your-alibaba-api-key",
	}
	alibabaStrategy, err := llmconnector.NewAlibabaStrategy(alibabaConfig)
	if err != nil {
		fmt.Println("Error setting up Alibaba strategy:", err)
		return
	}
	modelContext.SetEmbedStrategy(alibabaStrategy)

	// Use the strategies as needed... 
}
```

### Performing Chat Operations

```go
chatMessages := []llmconnector.ChatMessage{
	{Role: "user", Content: "Hello, how are you?"},
}
chatResponse, err := modelContext.Chat(ctx, chatMessages,
	llmconnector.WithChatModel("gpt-3.5-turbo"),
	llmconnector.WithTemperature(0.7),
)
if err != nil {
	fmt.Println("Error performing chat operation:", err)
	return
}
fmt.Println("Chat Response:", chatResponse.GetContent())
```

### Performing Embedding Operations

```go
texts := []string{"Hello world", "How are you?"}
embedResponse, err := modelContext.Embed(ctx, texts,
	llmconnector.WithEmbedModel("text-embedding-ada-002"),
)
if err != nil {
	fmt.Println("Error performing embedding operation:", err)
	return
}
fmt.Println("Embeddings:", embedResponse.GetEmbeddings())
```

### Customizing Options

You can customize the chat and embedding requests using various options:

```go
chatResponse, err := modelContext.Chat(ctx, chatMessages,
	llmconnector.WithChatModel("gpt-3.5-turbo"),
	llmconnector.WithTemperature(0.7),
	llmconnector.WithMaxTokens(800),
	llmconnector.WithTopP(0.9),
	llmconnector.WithStop([]string{"###"}),
)
```
For embedding operations:

```go
embedResponse, err := modelContext.Embed(ctx, texts,
	llmconnector.WithEmbedModel("text-embedding-ada-002"),
	llmconnector.WithEmbeddingType("document"),
)
```

### Advanced Configuration

`llmconnector` now supports advanced HTTP client configuration through the `github.com/simp-lee/gohttpclient` package. You can configure:

- Timeout
- Retries
- Rate Limiting
- Proxy
- Connection Pooling

These configurations are now part of the CommonConfig structure, which is shared across different providers:

```go
commonConfig := llmconnector.CommonConfig{
	Timeout:                30 * time.Second,
	Retries:                3,
	MaxNumRequestPerSecond: 10,
	MaxNumRequestPerLimit:  10,
	ProxyURL:               "http://proxy.example.com:8080",
	MaxIdleConns:           100,
	MaxConnsPerHost:        10,
	IdleConnTimeout:        90 * time.Second,
}
```

These configurations help in managing API rate limits, improving reliability with retries, and optimizing performance with connection pooling.

### Best Practices

- **API Key Security:** Never hardcode API keys in your source code. Use environment variables or secure configuration management.
- **Context Usage:** Always pass a context to Chat and Embed operations for proper cancellation and timeout handling.
- **Rate Limiting:** Be aware of the rate limits of the LLM providers you're using. Use the `MaxNumRequestPerSecond` 
  and `MaxNumRequestPerLimit` options to prevent rate limiting.
- **Error Handling:** Always check for errors returned by the `llmconnector` functions and handle them appropriately.
- **Proxy Usage:** If you're operating in an environment that requires a proxy, make sure to configure it correctly in the `CommonConfig`.

### Supported Language Models

- OpenAI: Supports chat and embedding operations using OpenAI's GPT models.
- Alibaba Cloud: Supports chat and embedding operations using Alibaba Cloud's NLP services.

Adding more models is straightforward. The `llmconnector` package provides simple interfaces (`ChatStrategy` and `EmbedStrategy`) for implementing new strategies.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License.