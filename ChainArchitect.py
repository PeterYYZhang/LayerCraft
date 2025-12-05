import os
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv
import json
import logging
import base64
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

# Load environment variables
load_dotenv()

class ChainArchitect:
    def __init__(self, output_dir: Optional[Path] = None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with open('prompts/chain_architect_sys.md', 'r') as f:
            self.system_prompt = f.read()
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        # Use provided output directory or default to "outputs"
        self.output_dir = output_dir if output_dir is not None else Path("outputs")
        self.output_dir.mkdir(exist_ok=True)

    def save_outputs(self, thinking_process: str, json_output: Dict[str, Any], prompt: str) -> None:
        """
        Save thinking process and JSON output to separate files.
        
        Args:
            thinking_process: The complete thinking process text
            json_output: The structured JSON output
            prompt: The user's prompt that generated these outputs
        """
        # Try to extract original prompt from scene structure if available
        original_prompt = None
        if isinstance(prompt, str) and "Here is the background generated" in prompt:
            # Extract scene structure from the prompt
            try:
                # First try to find the scene structure JSON
                scene_structure_start = prompt.find("{")
                scene_structure_end = prompt.rfind("}") + 1
                if scene_structure_start != -1 and scene_structure_end != -1:
                    scene_structure_str = prompt[scene_structure_start:scene_structure_end]
                    # Clean up the JSON string
                    scene_structure_str = scene_structure_str.replace("```json", "").replace("```", "").strip()
                    # Try to fix common JSON formatting issues
                    scene_structure_str = scene_structure_str.replace("'", '"')
                    import re
                    scene_structure_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', scene_structure_str)
                    
                    scene_structure = json.loads(scene_structure_str)
                    # Check for original_prompt in the scene structure
                    if isinstance(scene_structure, dict):
                        if "original_prompt" in scene_structure:
                            original_prompt = scene_structure["original_prompt"]
                        # If not found, try to extract from the text before the JSON
                        elif "Here is the background generated" in prompt:
                            # Look for the original prompt in the text before the JSON
                            text_before_json = prompt[:scene_structure_start].strip()
                            if text_before_json:
                                # Try to find the last sentence or phrase that looks like a prompt
                                sentences = text_before_json.split('.')
                                for sentence in reversed(sentences):
                                    sentence = sentence.strip()
                                    if sentence and len(sentence) > 10:  # Basic check for a reasonable prompt length
                                        original_prompt = sentence
                                        break
            except Exception as e:
                logger.warning(f"Error extracting original prompt: {str(e)}")
                # If JSON parsing fails, try to extract from text before the JSON
                try:
                    text_before_json = prompt[:prompt.find("{")].strip()
                    if text_before_json:
                        sentences = text_before_json.split('.')
                        for sentence in reversed(sentences):
                            sentence = sentence.strip()
                            if sentence and len(sentence) > 10:
                                original_prompt = sentence
                                break
                except Exception as e2:
                    logger.warning(f"Failed to extract prompt from text: {str(e2)}")
        
        # Use original prompt if found, otherwise use the provided prompt
        prompt_to_use = original_prompt if original_prompt else prompt
        
        # Create a safe filename from the prompt
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt_to_use[:50])
        
        # Create prompt-specific directory
        prompt_dir = self.output_dir / safe_prompt
        prompt_dir.mkdir(exist_ok=True)
        
        # Save thinking process
        thinking_file = prompt_dir / "chainarchitect_thinking_process.md"
        with open(thinking_file, 'w') as f:
            f.write(thinking_process)
        logger.info(f"Saved thinking process to {thinking_file}")
        
        # Save JSON output
        json_file = prompt_dir / "chainarchitect_object_placements.json"
        with open(json_file, 'w') as f:
            json.dump(json_output, f, indent=2)
        logger.info(f"Saved JSON output to {json_file}")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def add_message(self, role: str, content: Union[str, List[Dict[str, Any]]]):
        """
        Add a message to the conversation.
        content can be either a string (text message) or a list of content parts (text + images)
        """
        assert role in ["user", "assistant", "system"]
        
        if isinstance(content, str):
            # Handle text-only message
            self.messages.append({"role": role, "content": content})
        else:
            # Handle message with images
            self.messages.append({"role": role, "content": content})

    def _extract_json_parts(self, text: str) -> tuple[str, Dict[str, Any]]:
        """
        Extract JSON parts from text, handling the GPT response format.
        
        Args:
            text: The text containing JSON objects
            
        Returns:
            tuple: (remaining_text, json_output)
                - remaining_text: Text without the JSON parts
                - json_output: The JSON object found, or error dict if none found
        """
        # Find the JSON part after the thinking process
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        
        if json_start == -1 or json_end == -1 or json_start >= json_end:
            return text, {"error": "No valid JSON object found in response"}
            
        # Extract the JSON part and clean it up
        json_str = text[json_start:json_end]
        
        # Try to clean up the JSON string
        try:
            # Remove any markdown code block markers
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            # Try to parse the JSON
            json_output = json.loads(json_str)
            
            # Get the text before the JSON (thinking process)
            thinking_process = text[:json_start].strip()
            
            # Remove any markdown code block markers from thinking process
            thinking_process = thinking_process.replace("```json", "").replace("```", "").strip()
            
            return thinking_process, json_output
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            # Try to fix common JSON formatting issues
            try:
                # Replace single quotes with double quotes
                json_str = json_str.replace("'", '"')
                # Add quotes to unquoted property names
                import re
                json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
                json_output = json.loads(json_str)
                thinking_process = text[:json_start].strip()
                return thinking_process, json_output
            except Exception as e2:
                logger.error(f"Failed to fix JSON formatting: {str(e2)}")
                return text, {"error": f"Invalid JSON in response: {str(e)}"}

    def get_response(self, message: Union[str, List[Dict[str, Any]]]) -> str:
        """
        Get response from the model.
        message can be either a string (text message) or a list of content parts (text + images)
        """
        self.add_message("user", message)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=self.messages,
                max_tokens=10000,
                temperature=0.5
            ).choices[0].message.content

            # Extract thinking process and JSON output from response
            try:
                # Extract JSON parts and get remaining text
                thinking_process, json_output = self._extract_json_parts(response)
                
                # Validate required fields for ChainArchitect
                if "error" not in json_output:
                    required_fields = ["background_analysis", "region_analysis", "object_placements"]
                    missing_fields = [field for field in required_fields if field not in json_output]
                    if missing_fields:
                        json_output = {"error": f"Missing required fields in JSON: {', '.join(missing_fields)}"}

                # Get the prompt text if message is a list of content parts
                prompt_text = message if isinstance(message, str) else next((part["text"] for part in message if part["type"] == "text"), "unknown_prompt")
                
                # Save both outputs
                self.save_outputs(thinking_process, json_output, prompt_text)
            except Exception as e:
                logger.error(f"Error processing response: {str(e)}")
                json_output = {"error": f"Error processing response: {str(e)}"}

            # Add the response to the conversation history
            self.add_message("assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting response from model: {str(e)}")
            raise

    def reset(self):
        """Reset the conversation history while keeping the system prompt."""
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

    def add_image_message(self, text: str, image_paths: List[str]) -> None:
        """
        Add a message containing both text and images.
        
        Args:
            text: The text content of the message
            image_paths: List of paths to the images to include
        """
        content = [{"type": "text", "text": text}]
        
        for image_path in image_paths:
            if not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                continue
                
            base64_image = self.encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        self.add_message("user", content)


if __name__ == "__main__":
    # Initialize architect
    architect = ChainArchitect()

    print("Enter your prompts and image paths (type 'exit' or 'quit' to end):")
    while True:
        # Get text input
        print("\nEnter your text prompt:")
        text_input = input("> ")
        
        # Check for exit conditions
        if text_input.lower() in ['exit', 'quit']:
            break
            
        # Get image path
        print("Enter path to image (or press enter to skip):")
        image_path = input("> ").strip()
        
        # Add image if provided
        if image_path:
            architect.add_image_message(text_input, [image_path])
            response = architect.get_response("")
        else:
            response = architect.get_response(text_input)
            
        print("\nResponse:")
        print(response)

    print("Goodbye!")


