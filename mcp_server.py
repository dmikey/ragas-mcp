#!/usr/bin/env python3
"""
MCP Server for RAGAS Experimental Evaluation
Implements JSON-RPC 2.0 over stdio for proper MCP integration.
"""

import asyncio
import json
import sys
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import uuid
from datetime import datetime

# Disable logging during normal operation to avoid interfering with stdio communication
# Only enable logging for development/debugging
import os
if os.getenv('MCP_DEBUG'):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    logger = logging.getLogger(__name__)
else:
    # Disable all logging for production use
    logging.disable(logging.CRITICAL)
    logger = logging.getLogger(__name__)
    logger.disabled = True

# Global storage for experiments and metrics
experiments: Dict[str, Dict] = {}
metrics: Dict[str, Any] = {}
projects: Dict[str, Any] = {}

@dataclass
class MCPMessage:
    jsonrpc: str = "2.0"
    method: Optional[str] = None
    params: Optional[Dict] = None
    id: Optional[Any] = None
    result: Optional[Any] = None
    error: Optional[Dict] = None

class MCPServer:
    def __init__(self):
        self.tools = {
            "create_experiment": self.create_experiment,
            "setup_metric": self.setup_metric,
            "evaluate_single": self.evaluate_single,
            "evaluate_batch": self.evaluate_batch,
            "list_experiments": self.list_experiments,
            "get_experiment": self.get_experiment,
            "delete_experiment": self.delete_experiment,
            "help": self.help,
            "quick_start": self.quick_start,
        }
        self.resources = {
            "experiments": "List all experiments",
            "metrics": "List all metrics",
        }

    async def create_experiment(self, params: Dict) -> Dict:
        """Create a new experiment"""
        experiment_id = params.get("experiment_id", str(uuid.uuid4()))
        
        experiment = {
            "experiment_id": experiment_id,
            "project_config": params.get("project_config", {}),
            "metric_configs": params.get("metric_configs", []),
            "description": params.get("description", ""),
            "created_at": datetime.now().isoformat(),
            "status": "created"
        }
        
        experiments[experiment_id] = experiment
        if os.getenv('MCP_DEBUG'):
            logger.info(f"Created experiment: {experiment_id}")
        
        return {
            "experiment_id": experiment_id,
            "status": "created",
            "message": f"Experiment {experiment_id} created successfully",
            "created_at": experiment["created_at"]
        }

    async def setup_metric(self, params: Dict) -> Dict:
        """Setup a metric for an experiment"""
        experiment_id = params.get("experiment_id")
        metric_name = params.get("name")
        
        if not experiment_id or experiment_id not in experiments:
            raise Exception(f"Experiment {experiment_id} not found")
        
        metric = {
            "name": metric_name,
            "prompt": params.get("prompt"),
            "values": params.get("values", []),
            "llm_config": params.get("llm_config", {}),
            "experiment_id": experiment_id
        }
        
        metric_key = f"{experiment_id}_{metric_name}"
        metrics[metric_key] = metric
        
        return {
            "metric_name": metric_name,
            "experiment_id": experiment_id,
            "status": "configured",
            "message": f"Metric {metric_name} configured for experiment {experiment_id}"
        }

    async def evaluate_single(self, params: Dict) -> Dict:
        """Evaluate a single input"""
        experiment_id = params.get("experiment_id")
        metric_name = params.get("metric_name")
        
        if not experiment_id or experiment_id not in experiments:
            available_experiments = list(experiments.keys())
            error_msg = f"Experiment '{experiment_id}' not found."
            if available_experiments:
                error_msg += f" Available experiments: {', '.join(available_experiments)}"
            else:
                error_msg += " Use 'quick_start' to create a demo experiment."
            raise Exception(error_msg)
        
        metric_key = f"{experiment_id}_{metric_name}"
        if metric_key not in metrics:
            available_metrics = [k.split('_', 1)[1] for k in metrics.keys() if k.startswith(f"{experiment_id}_")]
            error_msg = f"Metric '{metric_name}' not found for experiment '{experiment_id}'."
            if available_metrics:
                error_msg += f" Available metrics: {', '.join(available_metrics)}"
            else:
                error_msg += " Use 'setup_metric' to configure metrics first."
            raise Exception(error_msg)
        
        # Validate required fields
        query = params.get("query")
        response = params.get("response")
        
        if not query:
            raise Exception("Missing 'query' parameter. This should be the user's question or input.")
        if not response:
            raise Exception("Missing 'response' parameter. This should be the AI's response to evaluate.")
        
        # Simulate evaluation (in real implementation, this would use RAGAS)
        evaluation_id = str(uuid.uuid4())
        
        # Generate more realistic mock scores based on metric type
        score = 0.85  # Default
        if "faithfulness" in metric_name.lower():
            score = 0.82
        elif "relevancy" in metric_name.lower():
            score = 0.88
        elif "precision" in metric_name.lower():
            score = 0.79
        elif "recall" in metric_name.lower():
            score = 0.91
        
        result = {
            "evaluation_id": evaluation_id,
            "experiment_id": experiment_id,
            "metric_name": metric_name,
            "query": query,
            "response": response,
            "expected_output": params.get("expected_output"),
            "score": score,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "interpretation": self._interpret_score(score, metric_name),
            "suggestions": self._get_improvement_suggestions(score, metric_name)
        }
        
        return result

    async def evaluate_batch(self, params: Dict) -> Dict:
        """Evaluate multiple inputs"""
        experiment_id = params.get("experiment_id")
        metric_name = params.get("metric_name")
        evaluations = params.get("evaluations", [])
        
        if not experiment_id or experiment_id not in experiments:
            available_experiments = list(experiments.keys())
            error_msg = f"Experiment '{experiment_id}' not found."
            if available_experiments:
                error_msg += f" Available experiments: {', '.join(available_experiments)}"
            else:
                error_msg += " Use 'quick_start' to create a demo experiment."
            raise Exception(error_msg)
        
        if not evaluations:
            raise Exception("No evaluation data provided. Please include an 'evaluations' array with query-response pairs.")
        
        results = []
        scores = []
        
        for i, eval_data in enumerate(evaluations):
            try:
                eval_params = {
                    "experiment_id": experiment_id,
                    "metric_name": metric_name,
                    **eval_data
                }
                result = await self.evaluate_single(eval_params)
                results.append(result)
                scores.append(result["score"])
            except Exception as e:
                if os.getenv('MCP_DEBUG'):
                    logger.error(f"Error evaluating item {i}: {e}")
                results.append({
                    "error": str(e),
                    "index": i,
                    "status": "failed"
                })
        
        # Calculate batch statistics
        valid_scores = [s for s in scores if isinstance(s, (int, float))]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
        return {
            "batch_id": str(uuid.uuid4()),
            "experiment_id": experiment_id,
            "metric_name": metric_name,
            "total_evaluations": len(evaluations),
            "successful_evaluations": len(valid_scores),
            "failed_evaluations": len(evaluations) - len(valid_scores),
            "average_score": round(avg_score, 3),
            "score_range": {
                "min": min(valid_scores) if valid_scores else None,
                "max": max(valid_scores) if valid_scores else None
            },
            "overall_interpretation": self._interpret_score(avg_score, metric_name),
            "results": results,
            "status": "completed"
        }

    async def list_experiments(self, params: Dict) -> Dict:
        """List all experiments"""
        if not experiments:
            return {
                "message": "No experiments found. Use 'create_experiment' or 'quick_start' to get started!",
                "experiments": [],
                "total": 0,
                "suggestion": "Try running 'quick_start' to create a demo experiment with common metrics."
            }
        
        experiment_list = []
        for exp_id, exp_data in experiments.items():
            experiment_list.append({
                "id": exp_id,
                "description": exp_data.get("description", "No description"),
                "created_at": exp_data.get("created_at"),
                "status": exp_data.get("status"),
                "metrics_count": len([k for k in metrics.keys() if k.startswith(f"{exp_id}_")])
            })
        
        return {
            "experiments": experiment_list,
            "total": len(experiments),
            "message": f"Found {len(experiments)} experiment(s). Use 'get_experiment' to see details for any experiment."
        }

    async def get_experiment(self, params: Dict) -> Dict:
        """Get experiment details"""
        experiment_id = params.get("experiment_id")
        if not experiment_id or experiment_id not in experiments:
            available_experiments = list(experiments.keys())
            error_msg = f"Experiment '{experiment_id}' not found."
            if available_experiments:
                error_msg += f" Available experiments: {', '.join(available_experiments)}"
            else:
                error_msg += " No experiments exist yet. Use 'create_experiment' or 'quick_start' to create one."
            raise Exception(error_msg)
        
        experiment = experiments[experiment_id].copy()
        
        # Add metrics information
        experiment_metrics = {k.split('_', 1)[1]: v for k, v in metrics.items() if k.startswith(f"{experiment_id}_")}
        experiment["configured_metrics"] = experiment_metrics
        experiment["metrics_count"] = len(experiment_metrics)
        
        # Add helpful next steps
        if not experiment_metrics:
            experiment["next_steps"] = [
                "Configure metrics with 'setup_metric'",
                "Popular metrics: faithfulness, answer_relevancy, context_precision"
            ]
        else:
            experiment["next_steps"] = [
                "Run evaluations with 'evaluate_single' or 'evaluate_batch'",
                f"Available metrics: {', '.join(experiment_metrics.keys())}"
            ]
        
        return experiment

    async def delete_experiment(self, params: Dict) -> Dict:
        """Delete an experiment"""
        experiment_id = params.get("experiment_id")
        if not experiment_id or experiment_id not in experiments:
            raise Exception(f"Experiment {experiment_id} not found")
        
        # Remove associated metrics
        metrics_to_remove = [k for k in metrics.keys() if k.startswith(f"{experiment_id}_")]
        for metric_key in metrics_to_remove:
            del metrics[metric_key]
        
        del experiments[experiment_id]
        
        return {
            "message": f"Experiment {experiment_id} deleted successfully",
            "deleted_metrics": len(metrics_to_remove)
        }

    async def handle_initialize(self, params: Dict) -> Dict:
        """Handle MCP initialize request"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": False
                },
                "resources": {
                    "subscribe": False,
                    "listChanged": False
                }
            },
            "serverInfo": {
                "name": "ragas-mcp-server",
                "version": "1.0.0"
            }
        }

    async def handle_tools_list(self, params: Dict) -> Dict:
        """Handle tools/list request"""
        tool_schemas = {
            "create_experiment": {
                "name": "create_experiment",
                "description": "Create a new RAGAS evaluation experiment. Use this when you want to start a new evaluation project or set up a new testing scenario.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "experiment_id": {
                            "type": "string",
                            "description": "Unique identifier for the experiment (optional, will generate if not provided)"
                        },
                        "description": {
                            "type": "string",
                            "description": "Human-readable description of what this experiment is testing"
                        },
                        "project_config": {
                            "type": "object",
                            "description": "Configuration settings for the project"
                        },
                        "metric_configs": {
                            "type": "array",
                            "description": "List of metrics to use in this experiment",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "config": {
                                        "type": "object",
                                        "description": "Configuration object for the metric"
                                    }
                                },
                                "required": ["name"]
                            }
                        }
                    },
                    "required": []
                }
            },
            "setup_metric": {
                "name": "setup_metric",
                "description": "Configure a specific evaluation metric for an experiment. Use this to define how you want to measure quality (e.g., relevance, faithfulness, answer correctness).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "experiment_id": {
                            "type": "string",
                            "description": "ID of the experiment to add this metric to"
                        },
                        "name": {
                            "type": "string",
                            "description": "Name of the metric (e.g., 'faithfulness', 'answer_relevancy', 'context_precision')"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Custom prompt template for this metric evaluation"
                        },
                        "values": {
                            "type": "array",
                            "description": "Possible values or scale for this metric",
                            "items": {
                                "type": "string"
                            }
                        },
                        "llm_config": {
                            "type": "object",
                            "description": "LLM configuration for metric evaluation"
                        }
                    },
                    "required": ["experiment_id", "name"]
                }
            },
            "evaluate_single": {
                "name": "evaluate_single",
                "description": "Evaluate a single query-response pair using a configured metric. Use this to test one specific interaction.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "experiment_id": {
                            "type": "string",
                            "description": "ID of the experiment containing the metric"
                        },
                        "metric_name": {
                            "type": "string",
                            "description": "Name of the metric to use for evaluation"
                        },
                        "query": {
                            "type": "string",
                            "description": "The user's question or input"
                        },
                        "response": {
                            "type": "string",
                            "description": "The AI's response to evaluate"
                        },
                        "expected_output": {
                            "type": "string",
                            "description": "The expected or ground truth answer (if available)"
                        }
                    },
                    "required": ["experiment_id", "metric_name", "query", "response"]
                }
            },
            "evaluate_batch": {
                "name": "evaluate_batch",
                "description": "Evaluate multiple query-response pairs at once using a configured metric. Use this for bulk evaluation of datasets.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "experiment_id": {
                            "type": "string",
                            "description": "ID of the experiment containing the metric"
                        },
                        "metric_name": {
                            "type": "string",
                            "description": "Name of the metric to use for evaluation"
                        },
                        "evaluations": {
                            "type": "array",
                            "description": "List of evaluation items, each containing query, response, and optionally expected_output",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "The user's question or input"},
                                    "response": {"type": "string", "description": "The AI's response to evaluate"},
                                    "expected_output": {"type": "string", "description": "The expected or ground truth answer (optional)"}
                                },
                                "required": ["query", "response"]
                            }
                        }
                    },
                    "required": ["experiment_id", "metric_name", "evaluations"]
                }
            },
            "list_experiments": {
                "name": "list_experiments",
                "description": "Show all available experiments. Use this to see what evaluation projects you have created.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            "get_experiment": {
                "name": "get_experiment",
                "description": "Get detailed information about a specific experiment, including its configuration and metrics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "experiment_id": {
                            "type": "string",
                            "description": "ID of the experiment to retrieve"
                        }
                    },
                    "required": ["experiment_id"]
                }
            },
            "delete_experiment": {
                "name": "delete_experiment",
                "description": "Delete an experiment and all its associated metrics. Use this to clean up experiments you no longer need.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "experiment_id": {
                            "type": "string",
                            "description": "ID of the experiment to delete"
                        }
                    },
                    "required": ["experiment_id"]
                }
            },
            "help": {
                "name": "help",
                "description": "Get help and guidance on using the RAGAS MCP server. Use this when you need information about available tools, workflows, or common metrics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Help topic: 'general', 'getting_started', or 'tools'",
                            "enum": ["general", "getting_started", "tools"]
                        }
                    },
                    "required": []
                }
            },
            "quick_start": {
                "name": "quick_start",
                "description": "Create a demo experiment with common RAGAS metrics to get started quickly. Perfect for first-time users who want to see how everything works.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        return {"tools": list(tool_schemas.values())}

    async def handle_tools_call(self, params: Dict) -> Dict:
        """Handle tools/call request"""
        tool_name = params.get("name")
        tool_arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            error_msg = self._generate_helpful_error(f"Unknown tool: {tool_name}")
            raise Exception(error_msg)
        
        # Validate parameters and provide suggestions
        suggestions = self._validate_and_suggest_params(tool_name, tool_arguments)
        
        try:
            result = await self.tools[tool_name](tool_arguments)
            
            # Add suggestions to successful results if applicable
            if suggestions:
                if isinstance(result, dict):
                    result["suggestions"] = suggestions
            
            # Format the response with better structure
            response_text = json.dumps(result, indent=2)
            
            # Add usage tips for first-time users
            if tool_name == "create_experiment" and result.get("status") == "created":
                response_text += "\n\n💡 Next steps:\n1. Use 'setup_metric' to configure evaluation metrics\n2. Use 'evaluate_single' or 'evaluate_batch' to run evaluations"
            elif tool_name == "setup_metric" and result.get("status") == "configured":
                response_text += "\n\n💡 Your metric is ready! Use 'evaluate_single' or 'evaluate_batch' to start evaluating."
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": response_text
                    }
                ]
            }
        except Exception as e:
            if os.getenv('MCP_DEBUG'):
                logger.error(f"Error executing tool {tool_name}: {e}")
            helpful_error = self._generate_helpful_error(str(e), tool_arguments)
            raise Exception(helpful_error)

    async def handle_resources_list(self, params: Dict) -> Dict:
        """Handle resources/list request"""
        resources = []
        for resource_name, description in self.resources.items():
            resources.append({
                "uri": f"ragas://{resource_name}",
                "name": resource_name,
                "description": description,
                "mimeType": "application/json"
            })
        
        return {"resources": resources}

    async def handle_resources_templates_list(self, params: Dict) -> Dict:
        """Handle resources/templates/list request"""
        # Return empty templates list as this server doesn't provide templates
        return {"resourceTemplates": []}

    async def handle_resources_read(self, params: Dict) -> Dict:
        """Handle resources/read request"""
        uri = params.get("uri", "")
        
        if uri == "ragas://experiments":
            # Return experiments data
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(list(experiments.values()), indent=2)
                    }
                ]
            }
        elif uri == "ragas://metrics":
            # Return metrics data
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json", 
                        "text": json.dumps(metrics, indent=2)
                    }
                ]
            }
        else:
            raise Exception(f"Resource not found: {uri}")

    def _suggest_tool_from_intent(self, user_message: str) -> Optional[str]:
        """Suggest the most appropriate tool based on user intent"""
        message_lower = user_message.lower()
        
        # Intent mapping based on common phrases
        intent_map = {
            "create": ["create_experiment"],
            "new": ["create_experiment"],
            "start": ["create_experiment"],
            "setup": ["setup_metric", "create_experiment"],
            "configure": ["setup_metric"],
            "metric": ["setup_metric"],
            "evaluate": ["evaluate_single", "evaluate_batch"],
            "test": ["evaluate_single", "evaluate_batch"],
            "check": ["evaluate_single", "get_experiment"],
            "single": ["evaluate_single"],
            "batch": ["evaluate_batch"],
            "multiple": ["evaluate_batch"],
            "list": ["list_experiments"],
            "show": ["list_experiments", "get_experiment"],
            "get": ["get_experiment"],
            "delete": ["delete_experiment"],
            "remove": ["delete_experiment"],
            "clean": ["delete_experiment"]
        }
        
        for keyword, tools in intent_map.items():
            if keyword in message_lower:
                return tools[0]  # Return the first/most relevant tool
        
        return None

    def _generate_helpful_error(self, error_msg: str, context: Dict = None) -> str:
        """Generate more helpful error messages with suggestions"""
        if "not found" in error_msg.lower():
            if context and "experiment_id" in str(context):
                return f"{error_msg}. Use 'list_experiments' to see available experiments, or 'create_experiment' to create a new one."
            elif "metric" in error_msg.lower():
                return f"{error_msg}. Use 'setup_metric' to configure this metric first."
        
        if "unknown tool" in error_msg.lower():
            return f"{error_msg}. Available tools: create_experiment, setup_metric, evaluate_single, evaluate_batch, list_experiments, get_experiment, delete_experiment."
        
        return error_msg

    def _validate_and_suggest_params(self, tool_name: str, params: Dict) -> Dict:
        """Validate parameters and suggest missing ones"""
        suggestions = {}
        
        if tool_name == "create_experiment":
            if not params.get("description"):
                suggestions["description"] = "Consider adding a description to make your experiment easier to identify later"
        
        elif tool_name == "setup_metric":
            if not params.get("experiment_id"):
                suggestions["experiment_id"] = "You need to specify which experiment this metric belongs to"
            if not params.get("name"):
                suggestions["name"] = "Popular RAGAS metrics include: faithfulness, answer_relevancy, context_precision, context_recall"
        
        elif tool_name in ["evaluate_single", "evaluate_batch"]:
            if not params.get("experiment_id"):
                suggestions["experiment_id"] = "Specify which experiment contains the metric you want to use"
            if not params.get("metric_name"):
                suggestions["metric_name"] = "Specify which metric to use for evaluation"
        
        return suggestions

    async def handle_message(self, message: Dict) -> Optional[Dict]:
        """Handle incoming MCP message"""
        try:
            method = message.get("method")
            params = message.get("params", {})
            message_id = message.get("id")
            
            if method == "initialize":
                result = await self.handle_initialize(params)
            elif method == "tools/list":
                result = await self.handle_tools_list(params)
            elif method == "tools/call":
                result = await self.handle_tools_call(params)
            elif method == "resources/list":
                result = await self.handle_resources_list(params)
            elif method == "resources/templates/list":
                result = await self.handle_resources_templates_list(params)
            elif method == "resources/read":
                result = await self.handle_resources_read(params)
            elif method == "notifications/initialized":
                # This is a notification, no response needed
                if os.getenv('MCP_DEBUG'):
                    logger.info("Client initialized")
                return None
            else:
                raise Exception(f"Unknown method: {method}")
            
            if message_id is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": result
                }
            
        except Exception as e:
            if os.getenv('MCP_DEBUG'):
                logger.error(f"Error handling message: {e}")
            if message.get("id") is not None:
                return {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    }
                }
        
        return None

    async def run(self):
        """Run the MCP server"""
        if os.getenv('MCP_DEBUG'):
            logger.info("Starting MCP RAGAS Server...")
        
        while True:
            try:
                # Read line from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Parse JSON message
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as e:
                    if os.getenv('MCP_DEBUG'):
                        logger.error(f"Invalid JSON: {e}")
                    continue
                
                # Handle message
                response = await self.handle_message(message)
                
                # Send response if needed
                if response:
                    print(json.dumps(response), flush=True)
                    
            except EOFError:
                break
            except Exception as e:
                if os.getenv('MCP_DEBUG'):
                    logger.error(f"Error in main loop: {e}")
                break
        
        if os.getenv('MCP_DEBUG'):
            logger.info("MCP Server shutting down...")

    async def help(self, params: Dict) -> Dict:
        """Provide help and guidance for using the RAGAS MCP server"""
        topic = params.get("topic", "general")
        
        help_content = {
            "general": {
                "title": "RAGAS MCP Server Help",
                "description": "This server helps you evaluate AI responses using RAGAS metrics.",
                "workflow": [
                    "1. Create an experiment with 'create_experiment'",
                    "2. Setup evaluation metrics with 'setup_metric'", 
                    "3. Run evaluations with 'evaluate_single' or 'evaluate_batch'",
                    "4. View results and manage experiments"
                ],
                "common_metrics": [
                    "faithfulness - How factually accurate is the response?",
                    "answer_relevancy - How relevant is the response to the question?",
                    "context_precision - How precise is the retrieved context?",
                    "context_recall - How complete is the retrieved context?"
                ]
            },
            "getting_started": {
                "title": "Quick Start Guide",
                "steps": [
                    "Use 'quick_start' to create a sample experiment",
                    "Or manually: create_experiment → setup_metric → evaluate_single/batch"
                ]
            },
            "tools": {
                "create_experiment": "Start a new evaluation project",
                "setup_metric": "Configure how to measure response quality",
                "evaluate_single": "Test one question-answer pair",
                "evaluate_batch": "Test multiple pairs at once",
                "list_experiments": "See all your experiments",
                "get_experiment": "View experiment details",
                "delete_experiment": "Remove an experiment"
            }
        }
        
        return help_content.get(topic, help_content["general"])

    async def quick_start(self, params: Dict) -> Dict:
        """Create a sample experiment with common metrics to get started quickly"""
        # Create a sample experiment
        experiment_params = {
            "experiment_id": "quick-start-demo",
            "description": "Quick start demo experiment with common RAGAS metrics",
            "project_config": {"type": "demo"}
        }
        
        experiment_result = await self.create_experiment(experiment_params)
        
        # Setup common metrics
        common_metrics = [
            {
                "name": "faithfulness",
                "prompt": "Evaluate how factually accurate this response is based on the given context."
            },
            {
                "name": "answer_relevancy", 
                "prompt": "Evaluate how relevant this response is to the original question."
            }
        ]
        
        metric_results = []
        for metric in common_metrics:
            metric_params = {
                "experiment_id": "quick-start-demo",
                "name": metric["name"],
                "prompt": metric["prompt"]
            }
            result = await self.setup_metric(metric_params)
            metric_results.append(result)
        
        return {
            "message": "Quick start demo created successfully!",
            "experiment": experiment_result,
            "metrics_configured": metric_results,
            "next_steps": [
                "Try: evaluate_single with experiment_id='quick-start-demo' and metric_name='faithfulness'",
                "Example evaluation: provide a query, response, and optionally expected_output"
            ],
            "example_usage": {
                "tool": "evaluate_single",
                "arguments": {
                    "experiment_id": "quick-start-demo",
                    "metric_name": "faithfulness",
                    "query": "What is the capital of France?",
                    "response": "The capital of France is Paris, which is located in the north-central part of the country.",
                    "expected_output": "Paris"
                }
            }
        }

    def _interpret_score(self, score: float, metric_name: str) -> str:
        """Provide human-readable interpretation of the score"""
        if score >= 0.9:
            quality = "Excellent"
        elif score >= 0.8:
            quality = "Good"
        elif score >= 0.7:
            quality = "Fair"
        elif score >= 0.6:
            quality = "Poor"
        else:
            quality = "Very Poor"
        
        metric_descriptions = {
            "faithfulness": "factual accuracy",
            "answer_relevancy": "relevance to the question",
            "context_precision": "precision of retrieved context",
            "context_recall": "completeness of retrieved context"
        }
        
        metric_desc = metric_descriptions.get(metric_name.lower(), "quality")
        return f"{quality} {metric_desc} (Score: {score:.2f}/1.0)"

    def _get_improvement_suggestions(self, score: float, metric_name: str) -> List[str]:
        """Provide suggestions for improvement based on the score and metric"""
        suggestions = []
        
        if score < 0.8:
            if "faithfulness" in metric_name.lower():
                suggestions.extend([
                    "Ensure all facts in the response are supported by the provided context",
                    "Remove or flag any information that cannot be verified from the source material",
                    "Consider using more precise language to avoid potential inaccuracies"
                ])
            elif "relevancy" in metric_name.lower():
                suggestions.extend([
                    "Make sure the response directly addresses the user's question",
                    "Remove tangential information that doesn't help answer the query",
                    "Consider restructuring the response to lead with the most relevant information"
                ])
            elif "precision" in metric_name.lower():
                suggestions.extend([
                    "Improve the quality of context retrieval",
                    "Filter out irrelevant retrieved documents",
                    "Consider adjusting retrieval parameters for better precision"
                ])
            elif "recall" in metric_name.lower():
                suggestions.extend([
                    "Ensure all relevant information is retrieved",
                    "Consider expanding the search query or retrieval scope",
                    "Review if important context pieces are being missed"
                ])
        
        if score >= 0.8:
            suggestions.append("Good performance! Consider testing with more diverse examples to validate consistency.")
        
        return suggestions

if __name__ == "__main__":
    server = MCPServer()
    asyncio.run(server.run())
