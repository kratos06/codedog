import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("complex_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ComplexExample:
    """
    A complex example class to demonstrate various Python features.
    
    This class includes multiple methods, error handling, and documentation
    to serve as a more complex test case for code evaluation.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ComplexExample class.
        
        Args:
            name: The name of this example
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.data = []
        logger.info(f"Initialized ComplexExample with name: {name}")
    
    def process_data(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of data dictionaries.
        
        Args:
            input_data: List of dictionaries to process
            
        Returns:
            Processed data as a list of dictionaries
            
        Raises:
            ValueError: If input_data is not a list
        """
        if not isinstance(input_data, list):
            logger.error("Input data must be a list")
            raise ValueError("Input data must be a list")
        
        result = []
        for item in input_data:
            try:
                processed_item = self._transform_item(item)
                result.append(processed_item)
                logger.debug(f"Processed item: {processed_item}")
            except Exception as e:
                logger.warning(f"Error processing item {item}: {str(e)}")
                # Skip this item and continue with the next
                continue
                
        self.data.extend(result)
        return result
    
    def _transform_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal method to transform a single data item.
        
        Args:
            item: Dictionary to transform
            
        Returns:
            Transformed dictionary
        """
        if not isinstance(item, dict):
            raise TypeError("Item must be a dictionary")
            
        # Apply transformations based on config
        result = item.copy()
        
        # Apply custom transformations if specified in config
        if "transformations" in self.config:
            for field, transform in self.config["transformations"].items():
                if field in result:
                    if transform == "uppercase" and isinstance(result[field], str):
                        result[field] = result[field].upper()
                    elif transform == "lowercase" and isinstance(result[field], str):
                        result[field] = result[field].lower()
                    elif transform == "double" and isinstance(result[field], (int, float)):
                        result[field] *= 2
        
        # Add metadata
        result["processed_by"] = self.name
        result["timestamp"] = self._get_timestamp()
        
        return result
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_results(self, filename: str) -> bool:
        """
        Save processed data to a JSON file.
        
        Args:
            filename: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.info(f"Saved results to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving results to {filename}: {str(e)}")
            return False
    
    def load_from_file(self, filename: str) -> bool:
        """
        Load data from a JSON file.
        
        Args:
            filename: Path to the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filename):
                logger.error(f"File not found: {filename}")
                return False
                
            with open(filename, 'r') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                logger.error(f"Invalid data format in {filename}")
                return False
                
            self.data = data
            logger.info(f"Loaded {len(data)} items from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {str(e)}")
            return False

def main():
    """Main function to demonstrate the ComplexExample class."""
    # Create an instance with configuration
    config = {
        "transformations": {
            "name": "uppercase",
            "value": "double"
        }
    }
    
    processor = ComplexExample("TestProcessor", config)
    
    # Sample data
    sample_data = [
        {"name": "item1", "value": 10, "category": "A"},
        {"name": "item2", "value": 20, "category": "B"},
        {"name": "item3", "value": 30, "category": "A"}
    ]
    
    # Process the data
    processed_data = processor.process_data(sample_data)
    
    # Print results
    print(f"Processed {len(processed_data)} items:")
    for item in processed_data:
        print(f"  - {item['name']}: {item['value']} ({item['category']})")
    
    # Save to file
    processor.save_results("processed_data.json")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
