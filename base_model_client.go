package llm

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"time"
)

// ChatMessage represents a chat message.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ModelClient is a generic interface for interacting with different types of models.
type ModelClient interface {
	Chat(messages []ChatMessage) (string, error)
}

// BaseModelClient is a base model client for interacting with different types of models.
type BaseModelClient struct {
	APIKey     string
	Endpoint   string
	ModelName  string
	HttpClient *http.Client
}

// HTTPClientConfig is a configuration for the HTTP client for the model client.
type HTTPClientConfig struct {
	Timeout            time.Duration
	MaxIdleConns       int
	IdleConnTimeout    time.Duration
	DisableCompression bool
	ProxyURL           string
}

// SetHTTPClient sets the HTTP client for the model client.
func (client *BaseModelClient) SetHTTPClient(config *HTTPClientConfig) {
	client.HttpClient = newHTTPClient(config)
}

func newHTTPClient(config *HTTPClientConfig) *http.Client {
	transport := &http.Transport{
		MaxIdleConns:       defaultIfZero(config.MaxIdleConns, 10),
		IdleConnTimeout:    defaultIfZeroDuration(config.IdleConnTimeout, 90*time.Second),
		DisableCompression: config.DisableCompression,
	}

	// Set the proxy URL if it's provided
	if config.ProxyURL != "" {
		if proxyURL, err := url.Parse(config.ProxyURL); err == nil {
			transport.Proxy = http.ProxyURL(proxyURL)
		} else {
			// Log the error if the proxy URL is not valid
			slog.Error("Failed to parse the proxy URL: "+config.ProxyURL, slog.Any("error", err))
		}
	}

	return &http.Client{
		Timeout:   config.Timeout,
		Transport: transport,
	}
}

func defaultIfZero(value, defaultValue int) int {
	if value == 0 {
		return defaultValue
	}
	return value
}

func defaultIfZeroDuration(value, defaultValue time.Duration) time.Duration {
	if value == 0 {
		return defaultValue
	}
	return value
}

// setDefaultHTTPClient sets the default HTTP client if it's not set.
func (client *BaseModelClient) setDefaultHTTPClient() {
	if client.HttpClient == nil {
		client.HttpClient = newHTTPClient(&HTTPClientConfig{
			Timeout:            120 * time.Second, // Increase the timeout to handle high latency or large data
			MaxIdleConns:       10,                // Increase the max idle connections to handle high concurrency
			IdleConnTimeout:    90 * time.Second,  // Decrease the idle connection timeout to release unused connections faster
			DisableCompression: false,             // Enable compression to reduce the data size for network transmission
		})
	}
}

// doRequest sends an HTTP request and returns the response.
func (client *BaseModelClient) doRequest(req *http.Request) (*http.Response, error) {
	var resp *http.Response
	var err error
	// Retry the request up to 3 times
	for i := 0; i < 3; i++ {
		if resp, err = client.HttpClient.Do(req); err == nil {
			return resp, nil
		}
		time.Sleep(time.Second * time.Duration(i))
	}
	return nil, err
}

// validateConfiguration validates the configuration.
func (client *BaseModelClient) validateConfiguration() error {
	if client.APIKey == "" {
		return errors.New("API key is required")
	}
	if client.Endpoint == "" {
		return errors.New("endpoint is required")
	}
	if client.ModelName == "" {
		return errors.New("model name is required")
	}
	if client.HttpClient == nil {
		return errors.New("HTTP client is not set")
	}
	return nil
}

// sendRequest sends a request to the model and returns the response.
func (client *BaseModelClient) sendRequest(body interface{}) (*http.Response, error) {
	// Set the default HTTP client if it's not already set
	client.setDefaultHTTPClient()

	// Validate the configuration
	if err := client.validateConfiguration(); err != nil {
		return nil, err
	}

	// Convert the body to JSON
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	// Build the request
	req, err := http.NewRequest("POST", client.Endpoint, bytes.NewBuffer(bodyBytes))
	if err != nil {
		return nil, err
	}

	// Set the headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+client.APIKey)

	// Send the request
	resp, err := client.doRequest(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send the request to %s: %w", client.Endpoint, err)
	}

	return resp, nil
}

// readResponseBody reads the response body.
func (client *BaseModelClient) readResponseBody(resp *http.Response) ([]byte, error) {
	// Check the response status code
	if resp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("server returned non-OK status code: %s, response: %s", resp.Status, string(respBytes))
	}

	return io.ReadAll(resp.Body)
}
