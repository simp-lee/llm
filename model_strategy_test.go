package llmconnector

import (
	"context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestModelContext_Chat(t *testing.T) {
	strategy := &MockChatStrategy{}
	ctx := NewModelContext()
	ctx.SetChatStrategy(strategy)

	messages := []ChatMessage{
		{Role: "user", Content: "Hello"},
	}

	resp, err := ctx.Chat(context.Background(), messages, WithChatModel("test-model"))
	require.NoError(t, err)
	require.NotNil(t, resp)

	chatResp, ok := resp.(*MockChatResponse)
	require.True(t, ok)
	assert.Equal(t, "Mock response", chatResp.GetContent())
}

func TestModelContext_Embed(t *testing.T) {
	strategy := &MockEmbedStrategy{}
	ctx := NewModelContext()
	ctx.SetEmbedStrategy(strategy)

	texts := []string{"text1", "text2"}

	resp, err := ctx.Embed(context.Background(), texts, WithEmbedModel("test-model"))
	require.NoError(t, err)
	require.NotNil(t, resp)

	embedResp, ok := resp.(*MockEmbedResponse)
	require.True(t, ok)
	assert.Equal(t, [][]float32{{0.1, 0.2, 0.3}}, embedResp.GetEmbeddings())
}

type MockChatStrategy struct{}

func (s *MockChatStrategy) Chat(ctx context.Context, chatMessages []ChatMessage, options *ChatOptions) (ChatResponse, error) {
	return &MockChatResponse{Content: "Mock response"}, nil
}

type MockChatResponse struct {
	Content string
}

func (r *MockChatResponse) GetContent() string {
	return r.Content
}

type MockEmbedStrategy struct{}

func (s *MockEmbedStrategy) Embed(ctx context.Context, texts []string, options *EmbedOptions) (EmbedResponse, error) {
	return &MockEmbedResponse{Embeddings: [][]float32{{0.1, 0.2, 0.3}}}, nil
}

type MockEmbedResponse struct {
	Embeddings [][]float32
}

func (r *MockEmbedResponse) GetEmbeddings() [][]float32 {
	return r.Embeddings
}
