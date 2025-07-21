# Data Structures Reference

This document provides comprehensive documentation of all data structures used in the MCP Tool Search System, including request/response formats, internal data models, and configuration schemas.

## Request Data Structures

### SearchRequest

**Description**: Standard request format for tool search operations.

```typescript
interface SearchRequest {
  query: string;                    // Natural language search query
  category?: string;                // Optional category filter
  tool_type?: string;               // Optional tool type filter  
  max_results?: number;             // Maximum results to return (1-50)
  include_examples?: boolean;       // Include usage examples
  search_mode?: SearchMode;         // Search algorithm to use
  context?: string;                 // Additional context information
  filters?: SearchFilters;          // Advanced filtering options
}

type SearchMode = 'semantic' | 'hybrid' | 'keyword';

interface SearchFilters {
  complexity_range?: {
    min: number;                    // Minimum complexity (0.0-1.0)
    max: number;                    // Maximum complexity (0.0-1.0)
  };
  has_examples?: boolean;           // Must have usage examples
  recent_only?: boolean;            // Only recently updated tools
  performance_tier?: 'fast' | 'balanced' | 'comprehensive';
}
```

**JSON Example**:
```json
{
  "query": "read configuration files safely",
  "category": "file_ops", 
  "max_results": 10,
  "include_examples": true,
  "search_mode": "hybrid",
  "context": "Python environment, error handling required",
  "filters": {
    "complexity_range": {"min": 0.2, "max": 0.8},
    "has_examples": true,
    "performance_tier": "balanced"
  }
}
```

### RecommendationRequest

**Description**: Request format for task-based tool recommendations.

```typescript
interface RecommendationRequest {
  task_description: string;         // Description of task to accomplish
  context?: string;                 // Environment/constraint context
  max_recommendations?: number;     // Maximum recommendations (1-10)
  include_alternatives?: boolean;   // Include alternative options
  complexity_preference?: ComplexityPreference;
  explain_reasoning?: boolean;      // Include detailed explanations
  user_preferences?: UserPreferences;
}

type ComplexityPreference = 'simple' | 'balanced' | 'powerful';

interface UserPreferences {
  preferred_categories?: string[];   // Preferred tool categories
  avoid_categories?: string[];       // Categories to avoid
  experience_level?: 'beginner' | 'intermediate' | 'expert';
  performance_priority?: number;     // Priority weight (0.0-1.0)
  reliability_priority?: number;     // Reliability weight (0.0-1.0)
}
```

### ComparisonRequest

**Description**: Request format for multi-tool comparison operations.

```typescript
interface ComparisonRequest {
  tool_ids: string[];               // Tool IDs to compare (2-10)
  comparison_criteria?: ComparisonCriterion[];
  include_decision_guidance?: boolean;
  comparison_context?: string;      // Context for comparison
  detail_level?: DetailLevel;       // Level of comparison detail
}

type ComparisonCriterion = 
  | 'functionality' 
  | 'usability' 
  | 'performance' 
  | 'reliability' 
  | 'complexity' 
  | 'documentation';

type DetailLevel = 'basic' | 'standard' | 'comprehensive';
```

## Response Data Structures

### SearchResponse

**Description**: Standard response format for search operations.

```typescript
interface SearchResponse {
  success: boolean;                 // Operation success status
  query: string;                    // Original query
  results: ToolSearchResult[];      // Search results
  total_results: number;            // Total number of results found
  execution_time: number;           // Query execution time (seconds)
  query_suggestions?: string[];     // Suggested query improvements
  category_breakdown?: CategoryBreakdown;
  timestamp: string;                // ISO 8601 timestamp
  metadata?: SearchMetadata;        // Additional search metadata
  error?: ErrorDetail;              // Error information if success=false
}

interface ToolSearchResult {
  tool_id: string;                  // Unique tool identifier
  name: string;                     // Display name
  description: string;              // Tool description
  similarity_score: number;         // Relevance score (0.0-1.0)
  confidence_level: ConfidenceLevel; // Confidence in match quality
  match_reasons: string[];          // Reasons for match
  parameters: ParameterInfo[];      // Tool parameters
  examples?: ToolExample[];         // Usage examples
  alternatives: string[];           // Alternative tool IDs
  complexity_score: number;         // Complexity rating (0.0-1.0)
  when_to_use?: string;            // Usage recommendations
  performance_characteristics?: PerformanceInfo;
  category: string;                 // Tool category
  tool_type: string;               // Tool type classification
  last_updated?: string;           // Last update timestamp
}

type ConfidenceLevel = 'low' | 'medium' | 'high';

interface CategoryBreakdown {
  [category: string]: number;       // Category name -> count
}

interface SearchMetadata {
  search_mode_used: SearchMode;     // Actual search mode used
  cache_hit: boolean;               // Whether result was cached
  algorithm_version: string;        // Search algorithm version
  total_catalog_size: number;       // Total tools in catalog
}
```

### RecommendationResponse

**Description**: Response format for tool recommendation requests.

```typescript
interface RecommendationResponse {
  success: boolean;
  task_description: string;         // Original task description
  recommendations: ToolRecommendation[];
  task_analysis: TaskAnalysis;      // Analysis of the task
  explanations?: string[];          // Detailed explanations
  metadata?: RecommendationMetadata;
  error?: ErrorDetail;
}

interface ToolRecommendation {
  tool: ToolSearchResult;           // Tool information
  recommendation_score: number;     // Overall recommendation score
  confidence_level: ConfidenceLevel;
  task_alignment: number;           // How well tool fits task (0.0-1.0)
  capability_match: number;         // Capability alignment score
  complexity_alignment: number;     // Complexity preference match
  recommendation_reasons: string[]; // Why this tool was recommended
  usage_guidance: string[];         // How to use effectively
  when_to_use: string;             // Optimal usage scenarios
  alternative_tools: string[];      // Alternative options
  risk_factors?: string[];          // Potential risks or limitations
  prerequisites?: string[];         // Requirements for usage
}

interface TaskAnalysis {
  task_category: string;            // Categorization of task
  complexity_level: ComplexityLevel;
  required_capabilities: string[];  // Needed capabilities
  optional_capabilities?: string[]; // Nice-to-have capabilities
  estimated_difficulty: number;     // Task difficulty (0.0-1.0)
  confidence: number;              // Analysis confidence (0.0-1.0)
  inferred_context: string;        // Context inferred from task
}

type ComplexityLevel = 'simple' | 'medium' | 'complex';
```

### ComparisonResponse

**Description**: Response format for tool comparison operations.

```typescript
interface ComparisonResponse {
  success: boolean;
  comparison: ToolComparison;
  metadata?: ComparisonMetadata;
  error?: ErrorDetail;
}

interface ToolComparison {
  tools: ToolComparisonResult[];    // Individual tool analyses
  summary: ComparisonSummary;       // Overall comparison summary
  criteria_used: ComparisonCriterion[];
  comparison_matrix?: ComparisonMatrix;
}

interface ToolComparisonResult {
  tool_id: string;
  tool_name: string;
  overall_score: number;            // Overall score (0.0-1.0)
  dimension_scores: DimensionScores; // Scores per criterion
  strengths: string[];              // Tool strengths
  weaknesses: string[];             // Tool limitations
  best_for: string[];              // Optimal use cases
  avoid_when: string[];            // Scenarios to avoid
  relative_ranking: number;         // Rank among compared tools
}

interface DimensionScores {
  functionality?: number;           // Feature richness (0.0-1.0)
  usability?: number;              // Ease of use (0.0-1.0)
  performance?: number;            // Speed and efficiency (0.0-1.0)
  reliability?: number;            // Stability and robustness (0.0-1.0)
  complexity?: number;             // Learning curve (0.0-1.0)
  documentation?: number;          // Documentation quality (0.0-1.0)
}

interface ComparisonSummary {
  recommended_choice: string;       // Best overall tool ID
  reasoning: string;               // Explanation for recommendation
  use_case_recommendations: {      // Specific use case guidance
    [use_case: string]: string;    // use_case -> recommended_tool_id
  };
  decision_factors: string[];      // Key decision criteria
  trade_offs: TradeOff[];         // Important trade-offs
}

interface TradeOff {
  factor: string;                  // Trade-off dimension
  description: string;             // Trade-off description
  affects_tools: string[];         // Which tools are affected
}
```

## Internal Data Structures

### ToolMetadata

**Description**: Internal representation of tool metadata.

```typescript
interface ToolMetadata {
  tool_id: string;
  name: string;
  description: string;
  category: string;
  tool_type: ToolType;
  version: string;
  author?: string;
  license?: string;
  repository_url?: string;
  documentation_url?: string;
  
  // Computed properties
  complexity_score: number;
  popularity_score: number;
  reliability_score: number;
  performance_tier: PerformanceTier;
  
  // Embeddings and search data
  embedding: number[];             // 384-dimension embedding
  search_keywords: string[];       // Extracted keywords
  semantic_tags: string[];         // Semantic classification tags
  
  // Parameter information
  parameters: ParameterDefinition[];
  return_schema?: JSONSchema;
  
  // Usage and examples
  examples: ToolExample[];
  usage_patterns: UsagePattern[];
  
  // Relationships
  similar_tools: string[];         // Similar tool IDs
  complementary_tools: string[];   // Complementary tools
  prerequisite_tools?: string[];   // Required prerequisite tools
  
  // Statistics and analytics
  usage_stats: UsageStatistics;
  
  // Metadata
  created_at: string;
  updated_at: string;
  indexed_at: string;
}

type ToolType = 'system' | 'custom' | 'third_party' | 'experimental';
type PerformanceTier = 'fast' | 'balanced' | 'comprehensive';
```

### ParameterDefinition

**Description**: Detailed parameter specification for tools.

```typescript
interface ParameterDefinition {
  name: string;                    // Parameter name
  type: ParameterType;             // Type specification
  required: boolean;               // Whether parameter is required
  description: string;             // Parameter description
  default_value?: any;             // Default value if optional
  validation_rules: ValidationRule[];
  examples: ParameterExample[];    // Usage examples
  constraints?: ParameterConstraints;
  deprecated?: boolean;            // Whether parameter is deprecated
  deprecation_message?: string;    // Deprecation guidance
}

type ParameterType = 
  | 'string' 
  | 'number' 
  | 'boolean' 
  | 'array' 
  | 'object' 
  | 'null'
  | 'union'
  | 'enum';

interface ValidationRule {
  type: ValidationType;
  value?: any;                     // Validation value
  message?: string;                // Error message
}

type ValidationType = 
  | 'min_length' 
  | 'max_length' 
  | 'pattern' 
  | 'min_value' 
  | 'max_value'
  | 'enum_values'
  | 'custom';

interface ParameterConstraints {
  min_length?: number;
  max_length?: number;
  pattern?: string;               // Regex pattern
  min_value?: number;
  max_value?: number;
  enum_values?: any[];
  depends_on?: string[];          // Parameter dependencies
}

interface ParameterExample {
  value: any;                     // Example value
  description: string;            // Example description
  use_case: string;              // When to use this value
}
```

### ToolExample

**Description**: Structure for tool usage examples.

```typescript
interface ToolExample {
  id: string;                     // Unique example ID
  title: string;                  // Example title
  description: string;            // What the example demonstrates
  code: string;                   // Example code
  language?: string;              // Programming language
  expected_output?: string;       // Expected result
  use_case: string;              // Scenario description
  complexity_level: ComplexityLevel;
  tags: string[];                // Example tags
  prerequisites?: string[];       // Requirements for example
  notes?: string;                // Additional notes
  related_examples?: string[];    // Related example IDs
}
```

### UsageStatistics

**Description**: Analytics and usage data for tools.

```typescript
interface UsageStatistics {
  total_uses: number;             // Total usage count
  successful_uses: number;        // Successful usage count
  failed_uses: number;           // Failed usage count
  success_rate: number;          // Success percentage
  
  // Performance metrics
  average_execution_time: number; // Average execution time (ms)
  median_execution_time: number;  // Median execution time (ms)
  p95_execution_time: number;     // 95th percentile execution time
  
  // Usage patterns
  common_parameters: ParameterUsage[];
  error_patterns: ErrorPattern[];
  user_feedback: UserFeedback[];
  
  // Temporal data
  daily_usage: TimeSeriesPoint[];
  usage_by_context: ContextUsage[];
  
  // Relationship data
  often_used_with: string[];      // Tools often used together
  alternative_selections: AlternativeSelection[];
  
  // Metadata
  last_updated: string;
  data_quality_score: number;     // Quality of statistics (0.0-1.0)
}

interface ParameterUsage {
  parameter_name: string;
  usage_frequency: number;        // How often this parameter is used
  common_values: ValueFrequency[];
}

interface ValueFrequency {
  value: any;
  frequency: number;
  percentage: number;
}

interface ErrorPattern {
  error_type: string;
  frequency: number;
  common_causes: string[];
  resolution_suggestions: string[];
}

interface UserFeedback {
  rating: number;                 // 1-5 star rating
  comment?: string;
  feedback_type: 'positive' | 'negative' | 'suggestion';
  timestamp: string;
  user_context?: string;
}

interface TimeSeriesPoint {
  timestamp: string;
  usage_count: number;
  success_rate: number;
}

interface ContextUsage {
  context: string;                // Context description
  usage_count: number;
  success_rate: number;
  common_errors?: string[];
}

interface AlternativeSelection {
  alternative_tool_id: string;
  selection_frequency: number;    // How often chosen over this tool
  reasons: string[];             // Reasons for alternative selection
}
```

## Configuration Data Structures

### SearchConfiguration

**Description**: System configuration for search behavior.

```typescript
interface SearchConfiguration {
  // Core search settings
  default_search_mode: SearchMode;
  max_results_limit: number;
  similarity_threshold: number;
  
  // Algorithm weights
  semantic_weight: number;
  keyword_weight: number;
  context_weight: number;
  
  // Performance settings
  cache_enabled: boolean;
  cache_size: number;
  cache_ttl: number;              // Time to live (seconds)
  timeout: number;               // Query timeout (ms)
  
  // Feature flags
  enable_learning: boolean;
  enable_feedback_collection: boolean;
  enable_analytics: boolean;
  enable_query_suggestions: boolean;
  
  // Quality settings
  min_confidence_threshold: number;
  result_diversity_factor: number;
  explanation_detail_level: DetailLevel;
  
  // Advanced settings
  embedding_model: string;
  vector_search_algorithm: string;
  ranking_algorithm: string;
  
  // Integration settings
  external_catalogs: ExternalCatalogConfig[];
  sync_intervals: SyncConfig;
}

interface ExternalCatalogConfig {
  catalog_id: string;
  type: 'api' | 'database' | 'file';
  endpoint: string;
  authentication: AuthConfig;
  sync_enabled: boolean;
  priority: number;              // Catalog priority (higher = more important)
}

interface AuthConfig {
  type: 'none' | 'api_key' | 'oauth' | 'basic';
  credentials: Record<string, string>;
}

interface SyncConfig {
  full_sync_interval: number;    // Full sync interval (hours)
  incremental_sync_interval: number; // Incremental sync (minutes)
  max_concurrent_syncs: number;
}
```

## Error Data Structures

### ErrorDetail

**Description**: Standardized error information structure.

```typescript
interface ErrorDetail {
  error_code: string;            // Unique error identifier
  message: string;               // Human-readable error message
  error_type: ErrorType;         // Classification of error
  context: ErrorContext;         // Error context information
  timestamp: string;             // When error occurred
  recovery_suggestions: string[]; // How to resolve the error
  related_errors?: string[];     // Related error codes
  debug_info?: DebugInfo;       // Additional debug information
}

type ErrorType = 
  | 'validation_error'
  | 'not_found_error'
  | 'system_error'
  | 'timeout_error'
  | 'permission_error'
  | 'rate_limit_error'
  | 'service_unavailable';

interface ErrorContext {
  operation: string;             // What operation was being performed
  parameters: Record<string, any>; // Parameters that caused error
  user_context?: string;         // User context if available
  system_state?: string;         // Relevant system state
  request_id?: string;          // Unique request identifier
}

interface DebugInfo {
  stack_trace?: string;
  internal_error_code?: string;
  component: string;             // Which component generated error
  version: string;              // Component version
  additional_data?: Record<string, any>;
}
```

## Validation Schemas

### Request Validation

JSON Schema for validating search requests:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["query"],
  "properties": {
    "query": {
      "type": "string",
      "minLength": 1,
      "maxLength": 500,
      "description": "Natural language search query"
    },
    "category": {
      "type": "string",
      "enum": ["file_ops", "web", "execution", "analysis", "data", "system"]
    },
    "max_results": {
      "type": "integer",
      "minimum": 1,
      "maximum": 50,
      "default": 10
    },
    "search_mode": {
      "type": "string",
      "enum": ["semantic", "hybrid", "keyword"],
      "default": "hybrid"
    },
    "filters": {
      "type": "object",
      "properties": {
        "complexity_range": {
          "type": "object",
          "properties": {
            "min": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "max": {"type": "number", "minimum": 0.0, "maximum": 1.0}
          },
          "additionalProperties": false
        },
        "has_examples": {"type": "boolean"},
        "recent_only": {"type": "boolean"}
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```

## Data Flow Diagrams

### Search Request Flow

```
User Request → Validation → Query Processing → Algorithm Selection → 
Search Execution → Result Ranking → Response Formatting → Cache Update → User Response
```

### Data Transformation Pipeline

```
Raw Tool Data → Metadata Extraction → Embedding Generation → 
Index Building → Quality Validation → Catalog Integration → Ready for Search
```

This comprehensive data structures reference ensures consistent data handling across all components of the MCP Tool Search System and provides clear contracts for API integration and system extension.