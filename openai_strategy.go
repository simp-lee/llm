package llmconnector

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/simp-lee/gohttpclient"
)

type OpenAIStrategy struct {
	chatClient  *gohttpclient.Client
	embedClient *gohttpclient.Client
	config      *Config
}

func NewOpenAIStrategy(config Config) (*OpenAIStrategy, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("OpenAI API key is required")
	}

	// Set default values if not provided
	if config.ChatURL == "" {
		config.ChatURL = "https://api.openai.com/v1/chat/completions"
	}
	if config.EmbedURL == "" {
		config.EmbedURL = "https://api.openai.com/v1/engines/text-similarity/embeddings"
	}

	// Use default common config if not set
	if config.CommonConfig == (CommonConfig{}) {
		config.CommonConfig = DefaultCommonConfig()
	}

	// Prepare the chat client
	chatClient, err := createClient(config.CommonConfig, config.APIKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create OpenAI chat client: %w", err)
	}

	// Prepare the embedding client
	embedClient, err := createClient(config.CommonConfig, config.APIKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create OpenAI embedding client: %w", err)
	}

	return &OpenAIStrategy{
		chatClient:  chatClient,
		embedClient: embedClient,
		config:      &config,
	}, nil
}

func (s *OpenAIStrategy) Chat(ctx context.Context, chatMessages []ChatMessage, options *ChatOptions) (ChatResponse, error) {
	request := map[string]interface{}{
		"model":    options.Model,
		"messages": chatMessages,
	}
	if options.Temperature != nil {
		request["temperature"] = *options.Temperature
	}
	if options.MaxTokens != nil {
		request["max_tokens"] = *options.MaxTokens
	}
	if options.TopP != nil {
		request["top_p"] = *options.TopP
	}
	if options.Stop != nil {
		request["stop"] = options.Stop
	}

	resp, err := s.chatClient.Post(ctx, s.config.ChatURL, request)
	if err != nil {
		return nil, fmt.Errorf("OpenAI chat request failed: %w", err)
	}

	var openAIResp OpenAIChatResponse
	if err := json.Unmarshal(resp, &openAIResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal OpenAI chat response: %w", err)
	}

	return &openAIResp, nil
}

type OpenAIChatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func (r *OpenAIChatResponse) GetContent() string {
	if len(r.Choices) > 0 {
		return r.Choices[0].Message.Content
	}
	return ""
}

func (s *OpenAIStrategy) Embed(ctx context.Context, texts []string, options *EmbedOptions) (EmbedResponse, error) {
	request := map[string]interface{}{
		"model": options.Model,
		"input": map[string]interface{}{
			"texts": texts,
		},
	}
	if options.EmbeddingType != "" {
		request["params"] = map[string]string{
			"text_type": options.EmbeddingType,
		}
	}

	resp, err := s.embedClient.Post(ctx, s.config.EmbedURL, request)
	if err != nil {
		return nil, fmt.Errorf("OpenAI embed request failed: %w", err)
	}

	var openAIResp OpenAIEmbedResponse
	if err := json.Unmarshal(resp, &openAIResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal OpenAI embed response: %w", err)
	}

	return &openAIResp, nil
}

type OpenAIEmbedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func (r *OpenAIEmbedResponse) GetEmbeddings() [][]float32 {
	embeddings := make([][]float32, len(r.Data))
	for i, data := range r.Data {
		embeddings[i] = data.Embedding
	}
	return embeddings
}
