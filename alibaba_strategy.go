package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/simp-lee/gohttpclient"
	"time"
)

type AlibabaConfig struct {
	APIKey   string
	ChatURL  string
	EmbedURL string
}

type AlibabaStrategy struct {
	chatClient  *gohttpclient.Client
	embedClient *gohttpclient.Client
	config      AlibabaConfig
}

func NewAlibabaStrategy(config AlibabaConfig) (*AlibabaStrategy, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("Alibaba API key is required")
	}

	// Set default base URLs if not set
	if config.ChatURL == "" {
		config.ChatURL = "https://dashscope.aliyuncs.com/api/v1/services/chat/completions"
	}
	if config.EmbedURL == "" {
		config.EmbedURL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding"
	}

	// Prepare the chat client
	chatClient := gohttpclient.NewClient(
		gohttpclient.WithTimeout(30*time.Second),
		gohttpclient.WithRetries(3),
	)
	chatClient.SetHeader("Authorization", fmt.Sprintf("Bearer %s", config.APIKey))
	chatClient.SetHeader("Content-Type", "application/json")

	// Prepare the embedding client
	embedClient := gohttpclient.NewClient(
		gohttpclient.WithTimeout(30*time.Second),
		gohttpclient.WithRetries(3),
	)
	embedClient.SetHeader("Authorization", fmt.Sprintf("Bearer %s", config.APIKey))
	embedClient.SetHeader("Content-Type", "application/json")

	return &AlibabaStrategy{
		chatClient:  chatClient,
		embedClient: embedClient,
		config:      config,
	}, nil
}

func (s *AlibabaStrategy) Chat(ctx context.Context, chatMessages []ChatMessage, options *ChatOptions) (ChatResponse, error) {
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
		return nil, fmt.Errorf("Alibaba chat request failed: %w", err)
	}

	var alibabaResp AlibabaChatResponse
	if err := json.Unmarshal(resp, &alibabaResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Alibaba chat response: %w", err)
	}

	return &alibabaResp, nil
}

type AlibabaChatResponse struct {
	Output struct {
		Text string `json:"text"`
	} `json:"output"`
}

func (r *AlibabaChatResponse) GetContent() string {
	return r.Output.Text
}

func (s *AlibabaStrategy) Embed(ctx context.Context, texts []string, options *EmbedOptions) (EmbedResponse, error) {
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
		return nil, fmt.Errorf("Alibaba embed request failed: %w", err)
	}

	var alibabaResp AlibabaEmbeddingResponse
	if err := json.Unmarshal(resp, &alibabaResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Alibaba embed response: %w", err)
	}

	return &AlibabaEmbedResponseWrapper{alibabaResp}, nil
}

type AlibabaEmbeddingResponse struct {
	Output struct {
		Embeddings []struct {
			TextIndex int       `json:"text_index"`
			Embedding []float32 `json:"embedding"`
		} `json:"embeddings"`
	} `json:"output"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
	RequestID string `json:"request_id"`
}

type AlibabaEmbedResponseWrapper struct {
	AlibabaEmbeddingResponse
}

func (r *AlibabaEmbedResponseWrapper) GetEmbeddings() [][]float32 {
	embeddings := make([][]float32, len(r.Output.Embeddings))
	for i, embedding := range r.Output.Embeddings {
		embeddings[i] = embedding.Embedding
	}
	return embeddings
}
