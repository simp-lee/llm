package llm

import (
	"context"
	"encoding/json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewOpenAIStrategy(t *testing.T) {
	config := OpenAIConfig{
		APIKey: "test-api-key",
	}

	strategy, err := NewOpenAIStrategy(config)
	require.NoError(t, err)
	require.NotNil(t, strategy)
}

func TestOpenAIStrategy_Chat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var request map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&request)
		require.NoError(t, err)

		assert.Equal(t, "test-model", request["model"])
		assert.Equal(t, []interface{}{map[string]interface{}{"role": "user", "content": "Hello"}}, request["messages"])

		response := `{"choices":[{"message":{"content":"Hi there"}}]}`
		w.Write([]byte(response))
	}))
	defer server.Close()

	config := OpenAIConfig{
		APIKey:  "test-api-key",
		ChatURL: server.URL,
	}

	strategy, err := NewOpenAIStrategy(config)
	require.NoError(t, err)
	require.NotNil(t, strategy)

	messages := []ChatMessage{
		{Role: "user", Content: "Hello"},
	}
	options := &ChatOptions{
		Model: "test-model",
	}

	resp, err := strategy.Chat(context.Background(), messages, options)
	require.NoError(t, err)
	require.NotNil(t, resp)

	chatResp, ok := resp.(*OpenAIChatResponse)
	require.True(t, ok)
	assert.Equal(t, "Hi there", chatResp.GetContent())
}

func TestOpenAIStrategy_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var request map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&request)
		require.NoError(t, err)

		assert.Equal(t, "test-model", request["model"])
		assert.Equal(t, []interface{}{"text1", "text2"}, request["input"].(map[string]interface{})["texts"])

		response := `{"data":[{"embedding":[0.1,0.2,0.3]}]}`
		w.Write([]byte(response))
	}))
	defer server.Close()

	config := OpenAIConfig{
		APIKey:   "test-api-key",
		EmbedURL: server.URL,
	}

	strategy, err := NewOpenAIStrategy(config)
	require.NoError(t, err)
	require.NotNil(t, strategy)

	texts := []string{"text1", "text2"}
	options := &EmbedOptions{
		Model: "test-model",
	}

	resp, err := strategy.Embed(context.Background(), texts, options)
	require.NoError(t, err)
	require.NotNil(t, resp)

	embedResp, ok := resp.(*OpenAIEmbedResponse)
	require.True(t, ok)
	assert.Equal(t, [][]float32{{0.1, 0.2, 0.3}}, embedResp.GetEmbeddings())
}

func TestOpenAIChatResponse_GetContent(t *testing.T) {
	resp := &OpenAIChatResponse{
		Choices: []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		}{
			{Message: struct {
				Content string `json:"content"`
			}{Content: "Hello, world!"}},
		},
	}

	assert.Equal(t, "Hello, world!", resp.GetContent())
}

func TestOpenAIChatResponse_GetContent_Empty(t *testing.T) {
	resp := &OpenAIChatResponse{}

	assert.Equal(t, "", resp.GetContent())
}

func TestOpenAIEmbedResponse_GetEmbeddings(t *testing.T) {
	resp := &OpenAIEmbedResponse{
		Data: []struct {
			Embedding []float32 `json:"embedding"`
		}{
			{Embedding: []float32{0.1, 0.2, 0.3}},
			{Embedding: []float32{0.4, 0.5, 0.6}},
		},
	}

	assert.Equal(t, [][]float32{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}, resp.GetEmbeddings())
}

func TestOpenAIEmbedResponse_GetEmbeddings_Empty(t *testing.T) {
	resp := &OpenAIEmbedResponse{}

	assert.Equal(t, [][]float32{}, resp.GetEmbeddings())
}
