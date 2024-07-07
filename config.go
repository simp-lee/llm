package llmconnector

import (
	"fmt"
	"github.com/simp-lee/gohttpclient"
	"net/url"
	"time"
)

// Config is a set of configuration options for all clients.
type Config struct {
	APIKey   string
	ChatURL  string
	EmbedURL string
	CommonConfig
}

// CommonConfig is a set of common configuration options for all clients.
type CommonConfig struct {
	Timeout time.Duration
	Retries int

	// the maximum number of requests allowed per second.
	MaxNumRequestPerSecond float64

	// the maximum number of requests allowed at once.
	MaxNumRequestPerLimit int

	ProxyURL        string
	MaxIdleConns    int
	MaxConnsPerHost int
	IdleConnTimeout time.Duration
}

// DefaultCommonConfig returns a default set of common configuration options.
func DefaultCommonConfig() CommonConfig {
	return CommonConfig{
		Timeout:                30 * time.Second,
		Retries:                3,
		MaxNumRequestPerSecond: 10,
		MaxNumRequestPerLimit:  10,
		MaxIdleConns:           100,
		MaxConnsPerHost:        10,
		IdleConnTimeout:        90 * time.Second,
	}
}

// createClient creates a new HTTP client with the given configuration options.
func createClient(config CommonConfig, apiKey string) (*gohttpclient.Client, error) {
	var options []gohttpclient.ClientOption

	if config.Timeout > 0 {
		options = append(options, gohttpclient.WithTimeout(config.Timeout))
	}

	if config.Retries > 0 {
		options = append(options, gohttpclient.WithRetries(config.Retries))
	}

	if config.MaxNumRequestPerLimit > 0 && config.MaxNumRequestPerSecond > 0 {
		options = append(options, gohttpclient.WithRateLimit(config.MaxNumRequestPerSecond, config.MaxNumRequestPerLimit))
	}

	if config.ProxyURL != "" {
		proxyURL, err := url.Parse(config.ProxyURL)
		if err != nil {
			return nil, fmt.Errorf("invalid proxy URL: %w", err)
		}
		options = append(options, gohttpclient.WithProxy(proxyURL))
	}

	if config.MaxIdleConns > 0 {
		options = append(options, gohttpclient.WithMaxIdleConns(config.MaxIdleConns))
	}

	if config.MaxConnsPerHost > 0 {
		options = append(options, gohttpclient.WithMaxConnsPerHost(config.MaxConnsPerHost))
	}

	if config.IdleConnTimeout > 0 {
		options = append(options, gohttpclient.WithIdleConnTimeout(config.IdleConnTimeout))
	}

	client := gohttpclient.NewClient(options...)
	client.SetHeader("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	client.SetHeader("Content-Type", "application/json")

	return client, nil
}
