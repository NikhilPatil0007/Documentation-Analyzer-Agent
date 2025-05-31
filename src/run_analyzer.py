"""
Script to run the documentation analyzer.
"""

from agent1_documentation_analyzer import DocumentationAnalyzer

def main():
    """Main execution function."""
    # URLs to analyze
    urls = [
        "https://help.moengage.com/hc/en-us/articles/11210398575508-All-Segments#h_01HD396C54709K6Z3BEKD8F8WF",
        "https://help.moengage.com/hc/en-us/articles/206169646-Create-Segments#h_01HE023CF3T3BXS40CD2NVQ96S"
    ]
    
    analyzer = DocumentationAnalyzer()
    
    for url in urls:
        print(f"\nAnalyzing URL: {url}")
        content = analyzer.fetch_content(url)
        
        if content and not content.startswith("Error"):
            report = analyzer.generate_report(url, content)
            
            # Create output directory if it doesn't exist
            import os
            os.makedirs("outputs", exist_ok=True)
            
            # Generate filename from URL
            filename = url.split("/")[-1].replace("-", "_")
            
            # Save JSON output
            json_file = f"outputs/{filename}_analysis.json"
            with open(json_file, "w", encoding="utf-8") as f:
                import json
                json.dump(report, f, indent=2)
            
            print(f"Analysis saved to: {json_file}")
        else:
            print(f"\nFailed to generate report for {url} due to content extraction error.")

if __name__ == "__main__":
    main() 