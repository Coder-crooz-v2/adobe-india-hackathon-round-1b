"""
Intelligent Document Analyst for Challenge 1b
Extracts and prioritizes relevant sections from document collections based on persona and job-to-be-done.

Approach:
1. Extract structured JSON from PDFs using font-based and content-based heading detection
2. Use TF-IDF vectorization for lightweight semantic similarity matching
3. Rank document sections based on relevance to the persona's job
4. Filter out irrelevant sections and extract detailed content for top-ranked sections
"""

import os
import json
import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
from collections import Counter
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class PDFStructureExtractor:
    """Extracts structured content from PDFs"""
    
    def _init_(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def extract_title(self, doc):
        """Extract the title from the first page by finding the largest font size"""
        first_page = doc[0]
        blocks = first_page.get_text("dict")["blocks"]
        
        # Find the maximum font size on the first page
        max_size = 0
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                if "spans" not in line:
                    continue
                for span in line["spans"]:
                    size = span.get("size", 0)
                    if size > max_size:
                        max_size = size
        
        if max_size == 0:
            return ""
        
        # Collect all text with the maximum font size
        title_parts = []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                if "spans" not in line:
                    continue
                
                line_text_parts = []
                has_max_size = False
                spans = sorted(line["spans"], key=lambda s: s.get("bbox", [0])[0])
                
                for span in spans:
                    if span.get("size", 0) == max_size:
                        has_max_size = True
                        line_text_parts.append(span.get("text", ""))
                
                if has_max_size and line_text_parts:
                    title_parts.append("".join(line_text_parts))
        
        if title_parts:
            full_title = " ".join(title_parts)
            full_title = re.sub(r'\s+', ' ', full_title).strip()
            return full_title
        
        return ""
    
    def is_decorative_line(self, text):
        """Detects decorative lines that should not be considered as headings"""
        text = text.strip()
        
        if len(text) < 3:
            return False
        
        dash_count = text.count('-')
        dot_count = text.count('.')
        underscore_count = text.count('_')
        equals_count = text.count('=')
        star_count = text.count('*')
        
        total_chars = len(text)
        decorative_chars = dash_count + dot_count + underscore_count + equals_count + star_count
        
        if decorative_chars > total_chars * 0.7:
            return True
        
        decorative_patterns = [
            r'^[-]{3,}$', r'^[.]{3,}$', r'^[_]{3,}$', 
            r'^[=]{3,}$', r'^[]{3,}$', r'^[-=._\s]{3,}$',
        ]
        
        for pattern in decorative_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def is_likely_link(self, text):
        """Detects if text is likely a URL or link"""
        text = text.strip().lower()
        url_patterns = [
            r'https?://', r'www\.', r'ftp://', r'mailto:', 
            r'\.com', r'\.org', r'\.net', r'\.edu', r'\.gov'
        ]
        
        for pattern in url_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def is_bulleted_heading(self, text):
        """Detects if text starts with bullet points or numbering patterns"""
        text = text.strip()
        
        if len(text) <= 3:
            return False
        
        patterns = [
            r'^\d+\.\s+\w+', r'^[a-z]\.\s+\w+', r'^[A-Z]\.\s+\w+',
            r'^\d+\.\d+\s+\w+', r'^\d+\.\d+\.\d+\s+\w+',
            r'^\([a-z]\)\s+\w+', r'^\([A-Z]\)\s+\w+', r'^\(\d+\)\s+\w+',
            r'^\d+\)\s+\w+', r'^[a-z]\)\s+\w+', r'^[A-Z]\)\s+\w+',
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def gather_spans(self, doc):
        """Gathers all text spans across the document"""
        items = []
        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    if "spans" not in line:
                        continue
                    
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if text and len(text) > 2:
                            items.append({
                                "text": text,
                                "size": span.get("size", 0),
                                "page": page_num
                            })
        
        return items
    
    def cluster_font_sizes(self, sizes):
        """Clusters unique font sizes and picks top 3 sizes for H1, H2, H3"""
        size_counts = Counter(sizes)
        most_abundant_size = size_counts.most_common(1)[0][0] if size_counts else None
        smallest_size = min(sizes) if sizes else None
        
        unique_sizes = sorted(set(sizes), reverse=True)
        
        # If all text has the same font size, we can't use font-based detection
        if len(unique_sizes) <= 1:
            return {}
        
        filtered_sizes = [size for size in unique_sizes 
                         if size != most_abundant_size and size != smallest_size]
        
        top_sizes = filtered_sizes[:3]
        level_map = {}
        for idx, size in enumerate(top_sizes, start=1):
            level_map[size] = idx
        return level_map
    
    def detect_headings(self, spans, level_map, title_text=""):
        """Build outline entries from spans and font-size-to-level mapping"""
        all_sizes = [s["size"] for s in spans]
        size_counts = Counter(all_sizes)
        most_abundant_size = size_counts.most_common(1)[0][0] if size_counts else None
        
        outline = []
        for s in spans:
            text = s["text"].strip()
            
            if text == title_text.strip():
                continue
            
            if s["size"] == most_abundant_size:
                continue
            
            if self.is_decorative_line(text) or self.is_likely_link(text):
                continue
                
            lvl = level_map.get(s["size"])
            is_font_heading = lvl is not None
            is_bullet_heading = self.is_bulleted_heading(text)
            
            if is_font_heading or is_bullet_heading:
                if is_bullet_heading and not is_font_heading:
                    heading_level = "H3"
                else:
                    heading_level = f"H{lvl}"
                
                outline.append({
                    "level": heading_level,
                    "text": text,
                    "page": s["page"]
                })
        
        return outline
    
    def extract_fallback_outline(self, doc):
        """Fallback method to extract outline when font-based detection fails"""
        outline = []
        
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                
                # Look for potential headings based on patterns
                is_heading = False
                level = "H3"  # Default level
                
                # Skip very generic words that aren't useful section titles
                skip_patterns = [
                    r'^(ingredients|instructions|directions|preparation|method|steps):?$',
                    r'^\d+[\.\)]\s*$',
                    r'^[a-z]+:$',
                    r'^\w{1,2}$'
                ]
                
                if any(re.match(pattern, line.lower()) for pattern in skip_patterns):
                    continue
                
                # Check for numbered sections (1. Item Name, 2. Item Name, etc.)
                if re.match(r'^\d+\.?\s+[A-Z]', line):
                    is_heading = True
                    level = "H2"
                elif (line.isupper() and 3 < len(line) < 50 and len(line.split()) <= 6 and
                      not re.match(r'^[A-Z\s]+:$', line)):
                    is_heading = True
                    level = "H1"
                elif (line.istitle() and 3 < len(line) < 80 and len(line.split()) <= 8 and
                      not line.endswith('.') and not line.endswith(',') and 
                      not line.endswith(':') and
                      not any(word in line.lower() for word in ['page', 'figure', 'table', 'www', 'http', 'see', 'the'])):
                    is_heading = True
                    level = "H2"
                elif (re.match(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)*$', line) and 
                      2 <= len(line.split()) <= 4 and 5 <= len(line) <= 30):
                    is_heading = True
                    level = "H3"
                
                if is_heading:
                    outline.append({"level": level, "text": line, "page": page_num})
        
        return outline[:25]  # Return more potential sections
    
    def extract_pdf_structure(self, pdf_path):
        """Extract structured data from a PDF file"""
        try:
            doc = fitz.open(pdf_path)
            title = self.extract_title(doc)
            spans = self.gather_spans(doc)
            
            if not spans:
                # Fallback: Extract all text and try to identify headings by content
                fallback_outline = self.extract_fallback_outline(doc)
                fallback_outline = self.enhance_outline_with_content(doc, fallback_outline)
                doc.close()
                return {"title": title, "outline": fallback_outline}
            
            sizes = [s["size"] for s in spans]
            level_map = self.cluster_font_sizes(sizes)
            
            # If font-based detection doesn't work, use fallback
            if not level_map:
                fallback_outline = self.extract_fallback_outline(doc)
                fallback_outline = self.enhance_outline_with_content(doc, fallback_outline)
                doc.close()
                return {"title": title, "outline": fallback_outline}
            
            outline = self.detect_headings(spans, level_map, title)
            
            # Enhance outline with content extraction
            outline = self.enhance_outline_with_content(doc, outline)
            
            doc.close()
            return {"title": title, "outline": outline}
            
        except Exception as e:
            return {"title": "", "outline": []}
    
    def enhance_outline_with_content(self, doc, outline):
        """Extract content that follows each heading"""
        enhanced_outline = []
        
        for section in outline:
            try:
                page_num = section.get('page', 1)
                if page_num > len(doc):
                    enhanced_outline.append(section)
                    continue
                
                page = doc[page_num - 1]  # Convert to 0-based index
                
                # Extract all text from the page
                page_text = page.get_text()
                
                # Find the position of this section's title in the page text
                section_title = section.get('text', '')
                
                if section_title and section_title in page_text:
                    title_pos = page_text.find(section_title)
                    content_after_title = page_text[title_pos + len(section_title):].strip()
                    
                    # Extract more content (up to 500 characters) after the title for better semantic matching
                    content_preview = content_after_title[:500].strip()
                    if content_preview:
                        section['content_preview'] = content_preview
                
                enhanced_outline.append(section)
                
            except Exception as e:
                enhanced_outline.append(section)
        
        return enhanced_outline


class SemanticAnalyzer:
    """Performs semantic analysis for document relevance ranking"""
    
    def _init_(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        """Preprocess text for semantic analysis"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in stopwords.words('english') and len(token) > 2]
        
        return ' '.join(tokens)
    
    def create_query_from_persona_and_job(self, persona, job_to_be_done):
        """Create a search query from persona and job description"""
        persona_text = f"role: {persona.get('role', '')}"
        job_text = f"task: {job_to_be_done.get('task', '')}"
        
        # Extract key terms and create expanded query
        combined_text = f"{persona_text} {job_text}"
        processed_text = self.preprocess_text(combined_text)
        
        # Extract key context clues from the job description
        job_desc = job_to_be_done.get('task', '').lower()
        
        # Add domain-specific terms based on persona role (generalized)
        role = persona.get('role', '').lower()
        domain_expansions = {
            'hr': ['human resources', 'employee', 'recruitment', 'onboarding', 'compliance', 'forms'],
            'researcher': ['research', 'methodology', 'analysis', 'data', 'study', 'literature'],
            'student': ['learn', 'study', 'education', 'course', 'assignment', 'tutorial'],
            'travel planner': ['travel', 'trip', 'destination', 'itinerary', 'hotels', 'attractions'],
            'salesperson': ['sales', 'customers', 'revenue', 'targets', 'leads', 'conversion'],
            'food contractor': ['menu', 'recipes', 'ingredients', 'cooking', 'food', 'catering'],
        }
        
        # Extract key terms from job description for context-aware expansion
        job_desc = job_to_be_done.get('task', '').lower()
        job_words = job_desc.split()
        
        # Dynamically add context terms based on what's mentioned in the job
        context_terms = []
        
        # Add terms that appear in the job description to emphasize them in search
        important_job_words = [word for word in job_words if len(word) > 3 and word not in ['with', 'from', 'that', 'this', 'have', 'will', 'been']]
        context_terms.extend(important_job_words)
        
        # Add role-specific terms
        for key, terms in domain_expansions.items():
            if key in role:
                processed_text += ' ' + ' '.join(terms)
                break
        
        # Add context terms
        if context_terms:
            processed_text += ' ' + ' '.join(context_terms)
        
        return processed_text
    
    def calculate_relevance_scores(self, sections, query):
        """Calculate relevance scores for document sections"""
        if not sections:
            return []
        
        # Prepare texts for vectorization
        section_texts = []
        for section in sections:
            # Include content preview if available for better semantic matching
            section_text = f"{section['text']} {section.get('level', '')}"
            if section.get('content_preview'):
                section_text += f" {section['content_preview']}"
            section_texts.append(self.preprocess_text(section_text))
        
        # Add query to texts for vectorization
        all_texts = section_texts + [query]
        
        try:
            # Vectorize all texts
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate similarity between each section and the query
            query_vector = tfidf_matrix[-1]
            section_vectors = tfidf_matrix[:-1]
            
            similarities = cosine_similarity(section_vectors, query_vector).flatten()
            
            # Apply sophisticated scoring with multiple factors
            for i, section in enumerate(sections):
                base_score = float(similarities[i])
                section_text = section['text'].lower()
                
                # Length penalty for very short sections (likely not substantial content)
                if len(section['text']) < 5:
                    base_score *= 0.5
                
                # Boost score for sections that appear to be substantial content
                # (reasonable length, not just generic words)
                if 5 <= len(section['text']) <= 50 and not section_text.endswith(':'):
                    base_score *= 1.2
                
                # Context-aware adjustments based on query content
                query_lower = query.lower()
                
                # General semantic boosting based on query-section similarity
                semantic_boost = 1.0
                
                # Extract key terms from query for semantic matching
                query_words = set(query_lower.split())
                section_words = set(section_text.split())
                
                # Calculate word overlap boost
                common_words = query_words & section_words
                if len(query_words) > 0:
                    overlap_ratio = len(common_words) / len(query_words)
                    semantic_boost = 1.0 + (overlap_ratio * 0.5)  # Up to 50% boost for high overlap
                
                # Domain-agnostic contextual matching
                # Look for any significant context words that appear in both query and section
                context_indicators = []
                for word in common_words:
                    if len(word) > 4:  # Focus on meaningful words
                        context_indicators.append(word)
                
                if context_indicators:
                    semantic_boost *= 1.2  # Additional boost for meaningful context matches
                
                base_score *= semantic_boost
                
                # General relevance adjustments
                relevance_boost = 1.0
                
                # Boost sections that contain terms directly mentioned in the persona or job
                query_terms = query_lower.split()
                significant_terms = [term for term in query_terms if len(term) > 3]
                
                for term in significant_terms:
                    if term in section_text:
                        relevance_boost *= 1.1  # Small boost for each relevant term match
                
                base_score *= relevance_boost
                
                # Penalize overly generic sections
                generic_penalty = 1.0
                generic_indicators = ['page', 'chapter', 'section', 'part', 'introduction', 'conclusion']
                if any(generic in section_text for generic in generic_indicators):
                    generic_penalty = 0.8
                
                base_score *= generic_penalty
                
                section['relevance_score'] = base_score
            
            # Sort sections by relevance score
            sections.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return sections
            
        except Exception as e:
            # Fallback: assign minimal scores
            for section in sections:
                section['relevance_score'] = 0.1
            return sections


class ContentExtractor:
    """Extracts detailed content from PDF sections"""
    
    def extract_section_content(self, pdf_path, page_num, section_title):
        """Extract detailed content for a specific section"""
        try:
            doc = fitz.open(pdf_path)
            
            if page_num > len(doc):
                doc.close()
                return ""
            
            page = doc[page_num - 1]  # Convert to 0-based index
            
            # Get all text blocks from the page
            blocks = page.get_text("dict")["blocks"]
            
            # Find the section and extract following content
            content_parts = []
            found_section = False
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    if "spans" not in line:
                        continue
                    
                    # Reconstruct line text
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span.get("text", "")
                    
                    line_text = line_text.strip()
                    
                    if not line_text:
                        continue
                    
                    # Check if this is our target section (more flexible matching)
                    title_words = section_title.lower().split()
                    line_words = line_text.lower().split()
                    
                    # Consider it a match if significant words overlap
                    if (section_title.lower() in line_text.lower() or 
                        line_text.lower() in section_title.lower() or
                        len(set(title_words) & set(line_words)) >= max(1, len(title_words) // 2)):
                        found_section = True
                        continue
                    
                    # If we found the section, collect following content
                    if found_section:
                        # Stop if we hit another potential heading
                        is_potential_heading = (
                            line_text.isupper() and len(line_text) < 50 or
                            line_text.istitle() and len(line_text) < 40 and not line_text.endswith('.') or
                            re.match(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)*$', line_text) and len(line_text.split()) <= 3
                        )
                        
                        if is_potential_heading and len(content_parts) > 0:
                            break
                        
                        # Add content that looks substantial
                        if len(line_text) > 10 and not line_text.endswith(':'):
                            content_parts.append(line_text)
                        
                        # Limit content length but be more generous
                        if len(' '.join(content_parts)) > 800:
                            break
            
            doc.close()
            
            # Return the most relevant content, prioritizing longer sentences
            if content_parts:
                # Sort by length and take the most substantial parts
                content_parts.sort(key=len, reverse=True)
                return ' '.join(content_parts[:4])  # Take top 4 most substantial parts
            
            return ""
            
        except Exception as e:
            return ""


class DocumentAnalyst:
    """Main class for intelligent document analysis"""
    
    def _init_(self):
        self.pdf_extractor = PDFStructureExtractor()
        self.semantic_analyzer = SemanticAnalyzer()
        self.content_extractor = ContentExtractor()
    
    def process_documents(self, input_json_path, pdf_directory):
        """Process documents according to input specification"""
        
        # Load input specification
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        persona = input_data.get('persona', {})
        job_to_be_done = input_data.get('job_to_be_done', {})
        documents = input_data.get('documents', [])
        
        # Extract structure from all PDFs
        all_sections = []
        document_structures = {}
        
        for doc_info in documents:
            filename = doc_info['filename']
            pdf_path = os.path.join(pdf_directory, filename)
            
            if not os.path.exists(pdf_path):
                continue
            
            structure = self.pdf_extractor.extract_pdf_structure(pdf_path)
            document_structures[filename] = structure
            
            # Add document info to each section
            for section in structure['outline']:
                section['document'] = filename
                section['document_title'] = structure['title']
                all_sections.append(section)
        
        # Create semantic query from persona and job
        query = self.semantic_analyzer.create_query_from_persona_and_job(persona, job_to_be_done)
        
        # Calculate relevance scores on all sections
        scored_sections = self.semantic_analyzer.calculate_relevance_scores(all_sections, query)
        
        # Sort by relevance score
        scored_sections.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Filter out sections with zero relevance scores (no semantic match)
        relevant_sections = [section for section in scored_sections if section.get('relevance_score', 0) > 0.0]
        
        # Select top sections (limit to 5 if we have more than 5 relevant sections)
        top_sections = relevant_sections[:5] if len(relevant_sections) >= 5 else relevant_sections
        
        # Extract detailed content for top sections
        subsection_analysis = []
        for section in top_sections:
            pdf_path = os.path.join(pdf_directory, section['document'])
            content = self.content_extractor.extract_section_content(
                pdf_path, section['page'], section['text']
            )
            if content:
                subsection_analysis.append({
                    "document": section['document'],
                    "refined_text": content,
                    "page_number": section['page']
                })
        
        # Create output
        output_data = {
            "metadata": {
                "input_documents": [doc['filename'] for doc in documents],
                "persona": persona.get('role', ''),
                "job_to_be_done": job_to_be_done.get('task', ''),
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_extracted": len(all_sections),
                "relevant_sections_found": len(top_sections)
            },
            "extracted_sections": [
                {
                    "document": section['document'],
                    "section_title": section['text'],
                    "importance_rank": i + 1,
                    "page_number": section['page'],
                    "relevance_score": round(section.get('relevance_score', 0), 4)
                }
                for i, section in enumerate(top_sections)
            ],
            "subsection_analysis": subsection_analysis
        }
        
        return output_data
    
    def run_analysis(self, input_json_path, pdf_directory, output_json_path):
        """Run complete analysis and save results"""
        try:
            results = self.process_documents(input_json_path, pdf_directory)
            
            # Save results
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            return results
            
        except Exception as e:
            raise


def main():
    """Main function to run the document analyst
    
    Usage:
    python main.py <input_json_path> <pdf_directory> <output_json_path>
    
    Example:
    python main.py "Collection 1/challenge1b_input.json" "Collection 1/PDFs" "output_collection1.json"
    """
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python main.py <input_json_path> <pdf_directory> <output_json_path>")
        print("Example: python main.py \"Collection 1/challenge1b_input.json\" \"Collection 1/PDFs\" \"output_collection1.json\"")
        sys.exit(1)
    
    input_json_path = sys.argv[1]
    pdf_directory = sys.argv[2]
    output_json_path = sys.argv[3]
    
    analyst = DocumentAnalyst()
    
    try:
        results = analyst.run_analysis(input_json_path, pdf_directory, output_json_path)
        print(f"Analysis complete! Found {len(results['extracted_sections'])} relevant sections.")
        print(f"Results saved to: {output_json_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if _name_ == "_main_":
    main()
