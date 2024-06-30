package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
)

// OpenAIChatModelClient is a client to access the OpenAI Chat API.
type OpenAIChatModelClient struct {
	BaseModelClient
	FrequencyPenalty float64
	PresencePenalty  float64
	N                int
	Stream           bool
	ResponseFormat   map[string]string
	Stop             []string
	Temperature      float64
	TopP             float64
	MaxTokens        int
}

// addOptionalParameters adds the optional parameters to the request body.
func (client *OpenAIChatModelClient) addOptionalParameters(body map[string]interface{}) {
	if client.FrequencyPenalty != 0 {
		body["frequency_penalty"] = client.FrequencyPenalty
	}
	if client.PresencePenalty != 0 {
		body["presence_penalty"] = client.PresencePenalty
	}
	if client.N != 0 {
		body["n"] = client.N
	}
	if client.Stream {
		body["stream"] = client.Stream
	}
	if client.ResponseFormat != nil {
		body["response_format"] = client.ResponseFormat
	}
	if client.Stop != nil {
		body["stop"] = client.Stop
	}
	if client.Temperature != 0 {
		body["temperature"] = client.Temperature
	}
	if client.TopP != 0 {
		body["top_p"] = client.TopP
	}
	if client.MaxTokens != 0 {
		body["max_tokens"] = client.MaxTokens
	}
}

// Chat sends a chat message to the model and returns the response.
func (client *OpenAIChatModelClient) Chat(messages []ChatMessage) (string, error) {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("Recovered from panic in OpenAIChatModelClient.Chat", slog.Any("error", r))
			// TODO: Handle the panic
		}
	}()

	// Check if the messages are empty
	if len(messages) == 0 {
		return "", errors.New("chat messages cannot be empty")
	}

	// Build the request body
	body := map[string]interface{}{
		"model":    client.ModelName,
		"messages": messages,
	}

	// Add optional parameters
	client.addOptionalParameters(body)

	// Send the request
	resp, err := client.sendRequest(body)
	if err != nil {
		return "", fmt.Errorf("failed to send the request to %s: %w", client.Endpoint, err)
	}
	defer resp.Body.Close()

	// Read the response body
	respBody, err := client.readResponseBody(resp)
	if err != nil {
		return "", err
	}

	// Parse the response
	return parseResponse(respBody)
}

// parseResponse parses the response and returns the generated message.
func parseResponse(response []byte) (string, error) {
	var responseMap map[string]interface{}
	if err := json.Unmarshal(response, &responseMap); err != nil {
		return "", fmt.Errorf("failed to parse the response: %w, response: %s", err, string(response))
	}

	choices, ok := responseMap["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("invalid response format: missing or empty 'choices', body: %s", string(response))
	}

	firstChoice, ok := choices[0].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("invalid response format: 'choices[0]' is not an object, body: %s", string(response))
	}

	message, ok := firstChoice["message"].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("invalid response format: 'choices[0].message' is not an object, body: %s", string(response))
	}

	generatedMessage, ok := message["content"].(string)
	if !ok {
		return "", fmt.Errorf("invalid response format: 'choices[0].message.content' is not a string, body: %s", string(response))
	}

	return generatedMessage, nil
}
