package llmconnector

import (
	"context"
	"encoding/json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewAlibabaStrategy(t *testing.T) {
	config := Config{
		APIKey: "test-api-key",
	}

	strategy, err := NewAlibabaStrategy(config)
	require.NoError(t, err)
	require.NotNil(t, strategy)
}

func TestAlibabaStrategy_Chat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var request map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&request)
		require.NoError(t, err)

		assert.Equal(t, "test-model", request["model"])
		assert.Equal(t, []interface{}{map[string]interface{}{"role": "user", "content": "Hello"}}, request["messages"])

		response := `{"output":{"text":"Hi there"}}`
		w.Write([]byte(response))
	}))
	defer server.Close()

	config := Config{
		APIKey:  "test-api-key",
		ChatURL: server.URL,
	}

	strategy, err := NewAlibabaStrategy(config)
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

	chatResp, ok := resp.(*AlibabaChatResponse)
	require.True(t, ok)
	assert.Equal(t, "Hi there", chatResp.GetContent())
}

func TestAlibabaStrategy_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer test-api-key", r.Header.Get("Authorization"))
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var request map[string]interface{}
		err := json.NewDecoder(r.Body).Decode(&request)
		require.NoError(t, err)

		assert.Equal(t, "test-model", request["model"])
		assert.Equal(t, []interface{}{"text1", "text2"}, request["input"].(map[string]interface{})["texts"])

		response := `{"output":{"embeddings":[{"text_index":0,"embedding":[0.1,0.2,0.3]}]}}`
		w.Write([]byte(response))
	}))
	defer server.Close()

	config := Config{
		APIKey:   "test-api-key",
		EmbedURL: server.URL,
	}

	strategy, err := NewAlibabaStrategy(config)
	require.NoError(t, err)
	require.NotNil(t, strategy)

	texts := []string{"text1", "text2"}
	options := &EmbedOptions{
		Model: "test-model",
	}

	resp, err := strategy.Embed(context.Background(), texts, options)
	require.NoError(t, err)
	require.NotNil(t, resp)

	embedResp, ok := resp.(*AlibabaEmbedResponseWrapper)
	require.True(t, ok)
	assert.Equal(t, [][]float32{{0.1, 0.2, 0.3}}, embedResp.GetEmbeddings())
}

func TestAlibabaChatResponse_GetContent(t *testing.T) {
	resp := &AlibabaChatResponse{
		Output: struct {
			Text string `json:"text"`
		}{
			Text: "Hello, world!",
		},
	}

	assert.Equal(t, "Hello, world!", resp.GetContent())
}

func TestAlibabaChatResponse_GetContent_Empty(t *testing.T) {
	resp := &AlibabaChatResponse{}

	assert.Equal(t, "", resp.GetContent())
}

func TestAlibabaEmbedResponseWrapper_GetEmbeddings(t *testing.T) {
	resp := &AlibabaEmbedResponseWrapper{
		AlibabaEmbeddingResponse: AlibabaEmbeddingResponse{
			Output: struct {
				Embeddings []struct {
					TextIndex int       `json:"text_index"`
					Embedding []float32 `json:"embedding"`
				} `json:"embeddings"`
			}{
				Embeddings: []struct {
					TextIndex int       `json:"text_index"`
					Embedding []float32 `json:"embedding"`
				}{
					{TextIndex: 0, Embedding: []float32{0.1, 0.2, 0.3}},
					{TextIndex: 1, Embedding: []float32{0.4, 0.5, 0.6}},
				},
			},
		},
	}

	assert.Equal(t, [][]float32{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}, resp.GetEmbeddings())
}

func TestAlibabaEmbedResponseWrapper_GetEmbeddings_Empty(t *testing.T) {
	resp := &AlibabaEmbedResponseWrapper{}

	assert.Equal(t, [][]float32{}, resp.GetEmbeddings())
}
