import os
import sqlite3
import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from enum import Enum
import chromadb
from chromadb.config import Settings


load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# STATE MANAGEMENT
# ==========================================

class WorkflowState(TypedDict):
    """State shared between agents in the workflow"""
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    user_query: str
    query_type: str
    sql_query: Optional[str]
    sql_results: Optional[Dict[str, Any]]
    knowledge_context: Optional[str]
    final_response: Optional[str]
    current_agent: str
    error: Optional[str]
    forecast_data: Optional[Dict[str, Any]]  # Added for forecast data

class ForecastMethod(Enum):
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ARIMA = "arima"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"

@dataclass
class ForecastResult:
    method: str
    predictions: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    accuracy_metrics: Dict[str, float]
    model_parameters: Dict[str, any]
    forecast_dates: List[str]


# ==========================================
# KNOWLEDGE RAG AGENT
# ==========================================

class KnowledgeAgent:
    """RAG-based Knowledge Agent using ChromaDB"""
    
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = None
        self._setup_vector_db()
        self._load_knowledge_base()
    
    def _setup_vector_db(self):
        """Initialize ChromaDB collection"""
        try:
            self.collection = self.chroma_client.get_collection("superstore_knowledge")
        except:
            self.collection = self.chroma_client.create_collection("superstore_knowledge")
    
    def _load_knowledge_base(self):
        """Load knowledge base into vector database"""
        knowledge_docs = [
            {
                "id": "schema_sales_data",
                "content": """
                Table: sales_data - Main transactional sales data
                Columns: row_id (PRIMARY KEY), order_id, order_date, ship_date, ship_mode, 
                customer_id, customer_name, segment, country, city, state, postal_code, region,
                product_id, category, sub_category, product_name, sales, quantity, discount, profit
                
                Business Rules:
                - order_date cannot be null, sales must be positive
                - discount between 0.0 and 1.0, profit can be negative
                - Historical data from superstore transactions
                """,
                "metadata": {"type": "schema", "table": "sales_data"}
            },
            {
                "id": "schema_forecasts",
                "content": """
                Table: base_forecasts - Generated demand forecasts
                Columns: forecast_id (PRIMARY KEY), product_id, region, segment, category,
                sub_category, forecast_date, forecasted_sales, forecasted_quantity, 
                forecast_method, confidence_interval_lower, confidence_interval_upper
                
                Forecast Methods: Moving Average, Seasonal Decomposition, Linear Trend
                Join patterns: f.product_id = s.product_id AND f.region = s.region AND f.forecast_date = s.order_date
                """,
                "metadata": {"type": "schema", "table": "base_forecasts"}
            },
            {
                "id": "business_rules",
                "content": """
                Customer Segments: Consumer (price-sensitive), Corporate (bulk orders), Home Office (medium orders)
                Regions: West and East typically outperform Central and South
                Seasonality: Q4 shows 20-30% higher sales due to holidays
                Product Categories: Furniture, Office Supplies, Technology
                Customer Value Tiers: High-Value (>$1000), Medium ($200-$1000), Low (<$200)
                """,
                "metadata": {"type": "business_rules"}
            },
            {
                "id": "sql_patterns",
                "content": """
                Common SQL Patterns for SQLite:
                - Time series: GROUP BY strftime('%Y-%m', order_date)
                - Top customers: ORDER BY SUM(sales) DESC LIMIT N
                - Regional analysis: GROUP BY region
                - Product performance: GROUP BY category, sub_category
                - Date filtering: WHERE order_date >= DATE('now', '-30 days')
                - Forecast joins: LEFT JOIN base_forecasts f ON s.product_id = f.product_id AND s.region = f.region
                """,
                "metadata": {"type": "sql_patterns"}
            }
        ]
        
        for doc in knowledge_docs:
            try:
                self.collection.add(
                    ids=[doc["id"]],
                    documents=[doc["content"]],
                    metadatas=[doc["metadata"]]
                )
            except Exception as e:
                logger.warning(f"Document {doc['id']} already exists: {e}")
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """Get relevant context for a query"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            if results["documents"]:
                context = "\n\n".join(results["documents"][0])
                return context
            return "No relevant context found."
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return "Error retrieving context."
    
    async def process_knowledge_query(self, state: WorkflowState) -> WorkflowState:
        """Process knowledge-based queries"""
        query = state["user_query"]
        context = self.get_context(query)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a database and business logic expert for a superstore dataset.
            
            Context from knowledge base:
            {context}
            
            Provide clear, accurate answers about the database schema, business rules, and data relationships.
            If asked about SQL queries, provide specific examples using the superstore schema."""),
            ("human", "{query}")
        ])
        
        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({"context": context, "query": query})
            
            state["knowledge_context"] = context
            state["final_response"] = response.content
            state["current_agent"] = "knowledge"
            
        except Exception as e:
            logger.error(f"Knowledge agent error: {e}")
            state["error"] = f"Knowledge agent error: {str(e)}"
        
        return state
    

# ==========================================
# FORECASTING AGENT
# ==========================================

class AdvancedForecastingAgent:
    """Fixed Advanced Forecasting Agent"""
    
    def __init__(self, llm: ChatGroq, db_path: str = "superstore.db"):
        self.llm = llm
        self.db_path = db_path
        self.scaler = StandardScaler()
    
    def load_sales_data_for_forecasting(self, product_id: str = None, region: str = None, 
                                      aggregation: str = 'daily') -> pd.DataFrame:
        """Load and prepare sales data with proper aggregation"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build base query
            query = """
            SELECT 
                order_date,
                product_id,
                region,
                category,
                sales,
                quantity
            FROM sales_data 
            WHERE 1=1
            """
            
            params = []
            if product_id:
                query += " AND product_id = ?"
                params.append(product_id)
            if region:
                query += " AND region = ?"
                params.append(region)
                
            query += " ORDER BY order_date"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                logger.warning("No data found for the specified criteria")
                return pd.DataFrame()
            
            # Convert date column
            df['order_date'] = pd.to_datetime(df['order_date'])
            
            # Aggregate data based on the specified period
            if aggregation == 'weekly':
                # Group by week
                df['period'] = df['order_date'].dt.to_period('W')
                aggregated = df.groupby('period').agg({
                    'sales': 'sum',
                    'quantity': 'sum'
                }).reset_index()
                aggregated['date'] = aggregated['period'].dt.to_timestamp()
                aggregated = aggregated.set_index('date')
                aggregated.columns = ['weekly_sales', 'weekly_quantity']
                
            elif aggregation == 'monthly':
                # Group by month
                df['period'] = df['order_date'].dt.to_period('M')
                aggregated = df.groupby('period').agg({
                    'sales': 'sum',
                    'quantity': 'sum'
                }).reset_index()
                aggregated['date'] = aggregated['period'].dt.to_timestamp()
                aggregated = aggregated.set_index('date')
                aggregated.columns = ['monthly_sales', 'monthly_quantity']
                
            else:  # daily
                # Group by day
                aggregated = df.groupby('order_date').agg({
                    'sales': 'sum',
                    'quantity': 'sum'
                }).reset_index()
                aggregated = aggregated.set_index('order_date')
                aggregated.columns = ['daily_sales', 'daily_quantity']
            
            # Fill missing dates with zeros for better forecasting
            if aggregation == 'daily':
                full_date_range = pd.date_range(
                    start=aggregated.index.min(),
                    end=aggregated.index.max(),
                    freq='D'
                )
                aggregated = aggregated.reindex(full_date_range, fill_value=0)
            
            logger.info(f"Loaded {len(aggregated)} {aggregation} data points")
            return aggregated
            
        except Exception as e:
            logger.error(f"Error loading forecasting data: {e}")
            return pd.DataFrame()
    
    def exponential_smoothing_forecast(self, df: pd.DataFrame, periods: int = 30, 
                                     aggregation: str = 'daily') -> ForecastResult:
        """Fixed Exponential Smoothing forecast"""
        if len(df) < 3:
            return self.simple_forecast_fallback(df, periods, aggregation)
        
        try:
            # Determine the sales column based on aggregation
            sales_column_map = {
                'daily': 'daily_sales',
                'weekly': 'weekly_sales', 
                'monthly': 'monthly_sales'
            }
            
            expected_column = sales_column_map.get(aggregation, 'daily_sales')
            
            # Find the actual sales column
            sales_column = None
            for col in df.columns:
                if 'sales' in col.lower():
                    sales_column = col
                    break
            
            if sales_column is None:
                # Use the first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    sales_column = numeric_cols[0]
                else:
                    return self.simple_forecast_fallback(df, periods, aggregation)
            
            logger.info(f"Using sales column: {sales_column}")
            sales_series = df[sales_column].fillna(0)
            
            # Remove any negative values
            sales_series = sales_series.clip(lower=0)
            
            # Use simple exponential smoothing for stability
            try:
                if len(sales_series) >= 10 and aggregation == 'daily':
                    # For daily data, try seasonal model with weekly seasonality
                    model = ExponentialSmoothing(
                        sales_series,
                        trend='add',
                        seasonal='add',
                        seasonal_periods=7
                    )
                else:
                    # For smaller datasets or non-daily data, use simple trend model
                    model = ExponentialSmoothing(
                        sales_series,
                        trend='add',
                        seasonal=None
                    )
                
                fitted_model = model.fit(optimized=True, use_brute=False)
                
            except Exception as e:
                logger.warning(f"Complex model failed ({e}), using simple exponential smoothing")
                # Fallback to simplest model
                model = ExponentialSmoothing(sales_series, trend=None, seasonal=None)
                fitted_model = model.fit()
            
            # Generate forecasts
            forecast = fitted_model.forecast(periods)
            predictions = [max(0, pred) for pred in forecast.tolist()]  # Ensure non-negative
            
            # Calculate confidence intervals
            if hasattr(fitted_model, 'resid') and len(fitted_model.resid) > 0:
                residuals = fitted_model.resid.dropna()
                if len(residuals) > 0:
                    std_error = np.std(residuals)
                else:
                    std_error = np.std(sales_series) * 0.2
            else:
                std_error = np.std(sales_series) * 0.2
            
            confidence_lower = [max(0, pred - 1.96 * std_error) for pred in predictions]
            confidence_upper = [pred + 1.96 * std_error for pred in predictions]
            
            # Generate forecast dates
            last_date = df.index[-1]
            forecast_dates = []
            
            for i in range(periods):
                if aggregation == 'weekly':
                    next_date = last_date + pd.Timedelta(weeks=i+1)
                elif aggregation == 'monthly':
                    next_date = last_date + pd.DateOffset(months=i+1)
                else:  # daily
                    next_date = last_date + pd.Timedelta(days=i+1)
                
                forecast_dates.append(next_date.strftime('%Y-%m-%d'))
            
            # Calculate accuracy metrics
            fitted_values = fitted_model.fittedvalues
            if len(fitted_values) > 0:
                actual_aligned = sales_series.iloc[-len(fitted_values):]
                mae = mean_absolute_error(actual_aligned, fitted_values)
                rmse = np.sqrt(mean_squared_error(actual_aligned, fitted_values))
            else:
                mae = std_error
                rmse = std_error
            
            return ForecastResult(
                method=f"Exponential Smoothing ({aggregation})",
                predictions=predictions,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                accuracy_metrics={"MAE": float(mae), "RMSE": float(rmse)},
                model_parameters={
                    "aggregation": aggregation,
                    "sales_column": sales_column,
                    "data_points": len(sales_series)
                },
                forecast_dates=forecast_dates
            )
            
        except Exception as e:
            logger.error(f"Exponential smoothing failed: {e}")
            return self.simple_forecast_fallback(df, periods, aggregation)
    
    def simple_forecast_fallback(self, df: pd.DataFrame, periods: int = 30, 
                                aggregation: str = 'daily') -> ForecastResult:
        """Simple moving average fallback"""
        try:
            if len(df) == 0:
                # Return zero forecast if no data
                forecast_dates = []
                base_date = datetime.now()
                for i in range(periods):
                    if aggregation == 'weekly':
                        next_date = base_date + timedelta(weeks=i+1)
                    elif aggregation == 'monthly':
                        next_date = base_date + timedelta(days=(i+1)*30) 
                    else:
                        next_date = base_date + timedelta(days=i+1)
                    forecast_dates.append(next_date.strftime('%Y-%m-%d'))
                
                return ForecastResult(
                    method="Zero Forecast (No Data)",
                    predictions=[0.0] * periods,
                    confidence_lower=[0.0] * periods,
                    confidence_upper=[0.0] * periods,
                    accuracy_metrics={"MAE": 0, "RMSE": 0},
                    model_parameters={"aggregation": aggregation},
                    forecast_dates=forecast_dates
                )
            
            # Find sales column
            sales_column = None
            for col in df.columns:
                if 'sales' in col.lower():
                    sales_column = col
                    break
            
            if sales_column is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    sales_column = numeric_cols[0]
                else:
                    raise ValueError("No sales column found")
            
            sales_series = df[sales_column].fillna(0)
            
            # Calculate moving average
            window = min(7, len(sales_series))
            recent_avg = sales_series.rolling(window=window, min_periods=1).mean().iloc[-1]
            
            if pd.isna(recent_avg):
                recent_avg = sales_series.mean()
                
            if pd.isna(recent_avg):
                recent_avg = 0
            
            predictions = [float(recent_avg)] * periods
            std_dev = float(sales_series.std()) if len(sales_series) > 1 else float(recent_avg * 0.2)
            
            confidence_lower = [max(0, pred - 1.96 * std_dev) for pred in predictions]
            confidence_upper = [pred + 1.96 * std_dev for pred in predictions]
            
            # Generate forecast dates
            last_date = df.index[-1]
            forecast_dates = []
            
            for i in range(periods):
                if aggregation == 'weekly':
                    next_date = last_date + pd.Timedelta(weeks=i+1)
                elif aggregation == 'monthly':
                    next_date = last_date + pd.DateOffset(months=i+1)
                else:
                    next_date = last_date + pd.Timedelta(days=i+1)
                
                forecast_dates.append(next_date.strftime('%Y-%m-%d'))
            
            return ForecastResult(
                method=f"Moving Average Fallback ({aggregation})",
                predictions=predictions,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                accuracy_metrics={"MAE": float(std_dev), "RMSE": float(std_dev)},
                model_parameters={"window": window, "sales_column": sales_column},
                forecast_dates=forecast_dates
            )
            
        except Exception as e:
            logger.error(f"Fallback forecast failed: {e}")
            # Ultimate fallback - return minimal forecast
            return ForecastResult(
                method="Error Fallback",
                predictions=[0.0] * periods,
                confidence_lower=[0.0] * periods,
                confidence_upper=[0.0] * periods,
                accuracy_metrics={"MAE": 0, "RMSE": 0},
                model_parameters={"error": str(e)},
                forecast_dates=[(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                              for i in range(periods)]
            )
    
    def extract_parameters_from_query(self, query: str) -> Dict[str, Any]:
        """Extract forecasting parameters from user query"""
        query_lower = query.lower()
        
        # Default parameters
        params = {
            'product_id': None,
            'region': None,
            'periods': 30,
            'aggregation': 'daily'
        }
        
        # Extract aggregation type
        if any(word in query_lower for word in ['week', 'weekly']):
            params['aggregation'] = 'weekly'
            params['periods'] = 4  # Default to 4 weeks
        elif any(word in query_lower for word in ['month', 'monthly']):
            params['aggregation'] = 'monthly'
            params['periods'] = 3  # Default to 3 months
        elif any(word in query_lower for word in ['quarter', 'quarterly']):
            params['aggregation'] = 'daily'
            params['periods'] = 90  # 3 months in days
        
        # Extract specific period numbers
        import re
        
        # Look for "next X days/weeks/months"
        period_match = re.search(r'next\s+(\d+)\s+(day|week|month)', query_lower)
        if period_match:
            num = int(period_match.group(1))
            unit = period_match.group(2)
            if unit == 'day':
                params['periods'] = num
                params['aggregation'] = 'daily'
            elif unit == 'week':
                params['periods'] = num
                params['aggregation'] = 'weekly'
            elif unit == 'month':
                params['periods'] = num
                params['aggregation'] = 'monthly'
        
        # Extract region
        regions = ['west', 'east', 'central', 'south']
        for region in regions:
            if region in query_lower:
                params['region'] = region.title()
                break
        
        # Extract specific products (basic approach)
        if 'furniture' in query_lower:
            params['category'] = 'Furniture'
        elif 'technology' in query_lower:
            params['category'] = 'Technology'
        elif 'office supplies' in query_lower:
            params['category'] = 'Office Supplies'
        
        return params
    
    async def generate_forecast_analysis(self, product_id: str = None, region: str = None, 
                                        periods: int = 30, aggregation: str = 'daily') -> Dict[str, Any]:
        """Main forecasting function"""
        try:
            logger.info(f"Generating forecast: product={product_id}, region={region}, periods={periods}, aggregation={aggregation}")
            
            # Load data
            df = self.load_sales_data_for_forecasting(product_id, region, aggregation)
            
            if len(df) == 0:
                return {
                    "success": False,
                    "error": f"No sales data found for the specified criteria (product={product_id}, region={region})"
                }
            
            # Generate forecast
            forecast_result = self.exponential_smoothing_forecast(df, periods, aggregation)
            
            # Calculate summary statistics
            if len(df) > 0:
                sales_col = None
                for col in df.columns:
                    if 'sales' in col.lower():
                        sales_col = col
                        break
                
                if sales_col:
                    avg_historical = float(df[sales_col].mean())
                    total_historical = float(df[sales_col].sum())
                else:
                    avg_historical = 0
                    total_historical = 0
            else:
                avg_historical = 0
                total_historical = 0
            
            # Generate business explanation
            context = f"Forecasting for Product: {product_id or 'All'}, Region: {region or 'All'}, Aggregation: {aggregation}"
            explanation = await self.generate_business_explanation(forecast_result, context, df)
            
            return {
                "success": True,
                "forecast": forecast_result,
                "business_explanation": explanation,
                "data_summary": {
                    "data_points": len(df),
                    "date_range": f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}" if len(df) > 0 else "No data",
                    "average_sales": avg_historical,
                    "total_historical_sales": total_historical,
                    "aggregation": aggregation
                }
            }
            
        except Exception as e:
            logger.error(f"Forecasting analysis error: {e}")
            return {
                "success": False,
                "error": f"Forecasting failed: {str(e)}"
            }
    
    async def generate_business_explanation(self, forecast_result: ForecastResult, 
                                          context: str, historical_data: pd.DataFrame) -> str:
        """Generate business explanation using LLM"""
        try:
            # Calculate summary statistics
            avg_prediction = float(np.mean(forecast_result.predictions))
            
            if len(historical_data) > 0:
                # Find sales column
                sales_col = None
                for col in historical_data.columns:
                    if 'sales' in col.lower():
                        sales_col = col
                        break
                
                if sales_col:
                    historical_avg = float(historical_data[sales_col].mean())
                    growth_rate = ((avg_prediction - historical_avg) / historical_avg) * 100 if historical_avg > 0 else 0
                else:
                    historical_avg = 0
                    growth_rate = 0
            else:
                historical_avg = 0
                growth_rate = 0
            
            forecast_summary = {
                "method": forecast_result.method,
                "forecast_period": f"{len(forecast_result.predictions)} periods",
                "predicted_average": f"${avg_prediction:.2f}",
                "historical_average": f"${historical_avg:.2f}",
                "growth_rate": f"{growth_rate:.1f}%",
                "confidence_range": f"${np.mean(forecast_result.confidence_lower):.2f} - ${np.mean(forecast_result.confidence_upper):.2f}",
                "accuracy_rmse": forecast_result.accuracy_metrics.get('RMSE', 0)
            }
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a business analyst explaining sales forecasts to stakeholders.
                
                Provide insights on:
                1. What the forecast means for business planning
                2. Growth/decline trends and implications  
                3. Confidence level and reliability
                4. Actionable recommendations
                5. Key risks and opportunities
                
                Context: {context}
                Keep explanations business-focused and actionable."""),
                ("human", "Explain this sales forecast: {forecast_summary}")
            ])
            
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "context": context,
                "forecast_summary": str(forecast_summary)
            })
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating business explanation: {e}")
            return f"Forecast shows expected trend. Detailed explanation unavailable due to: {str(e)}"
    
    async def process_forecasting_query(self, state: WorkflowState) -> WorkflowState:
        """Process forecasting queries in the workflow"""
        try:
            query = state["user_query"]
            
            # Extract parameters from query
            params = self.extract_parameters_from_query(query)
            
            logger.info(f"Extracted parameters: {params}")
            
            # Generate forecast
            result = await self.generate_forecast_analysis(
                product_id=params.get('product_id'),
                region=params.get('region'),
                periods=params.get('periods', 30),
                aggregation=params.get('aggregation', 'daily')
            )
            
            if result["success"]:
                forecast = result["forecast"]
                
                # Format the response
                aggregation = params.get('aggregation', 'daily')
                periods = params.get('periods', 30)
                
                if aggregation == 'weekly':
                    period_label = f"{periods} weeks"
                    avg_label = "Average Weekly Prediction"
                elif aggregation == 'monthly':
                    period_label = f"{periods} months"  
                    avg_label = "Average Monthly Prediction"
                else:
                    period_label = f"{periods} days"
                    avg_label = "Average Daily Prediction"
                
                avg_pred = np.mean(forecast.predictions)
                avg_conf_lower = np.mean(forecast.confidence_lower)
                avg_conf_upper = np.mean(forecast.confidence_upper)
                
                response = f"""📈 **Sales Forecast Analysis**

**Method**: {forecast.method}
**Period**: {period_label}
**{avg_label}**: ${avg_pred:.2f}
**Confidence Range**: ${avg_conf_lower:.2f} - ${avg_conf_upper:.2f}

📊 **Data Summary**:
- Historical data points: {result['data_summary']['data_points']}
- Date range: {result['data_summary']['date_range']}
- Historical average: ${result['data_summary']['average_sales']:.2f}
- Total historical sales: ${result['data_summary']['total_historical_sales']:.2f}

💡 **Business Insights**:
{result['business_explanation']}

🎯 **Model Performance**:
- RMSE: {forecast.accuracy_metrics.get('RMSE', 0):.2f}
- MAE: {forecast.accuracy_metrics.get('MAE', 0):.2f}
- Method: {forecast.method}

📅 **First 5 Forecast Points**:"""
                
                # Add first 5 forecast points
                for i in range(min(5, len(forecast.predictions))):
                    response += f"\n   {forecast.forecast_dates[i]}: ${forecast.predictions[i]:.2f}"
                
                if len(forecast.predictions) > 5:
                    response += f"\n   ... and {len(forecast.predictions) - 5} more forecast points"
                
                state["final_response"] = response
                state["forecast_data"] = {
                    "predictions": forecast.predictions,
                    "dates": forecast.forecast_dates,
                    "confidence_lower": forecast.confidence_lower,
                    "confidence_upper": forecast.confidence_upper,
                    "aggregation": aggregation,
                    "method": forecast.method
                }
            else:
                state["final_response"] = f"❌ Forecasting Error: {result['error']}"
                
            state["current_agent"] = "forecasting"
            
        except Exception as e:
            logger.error(f"Forecasting query processing error: {e}")
            state["error"] = f"Forecasting error: {str(e)}"
            state["final_response"] = f"❌ Forecasting failed: {str(e)}"
        
        return state


# ==========================================
# SQL DATABASE AGENT
# ==========================================

class SQLAgent:
    """SQL Database Agent using LangGraph"""
    
    def __init__(self, llm: ChatGroq, db_path: str = "superstore.db"):
        self.llm = llm
        self.db_path = db_path
        self.knowledge_agent = None
    
    def set_knowledge_agent(self, knowledge_agent: KnowledgeAgent):
        """Set reference to knowledge agent"""
        self.knowledge_agent = knowledge_agent
    
    def get_data_date_range(self) -> Dict[str, str]:
        """Get the actual date range of data in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT MIN(order_date) as min_date, MAX(order_date) as max_date FROM sales_data")
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    "min_date": result[0] if result[0] else "Unknown",
                    "max_date": result[1] if result[1] else "Unknown"
                }
            else:
                return {"min_date": "Unknown", "max_date": "Unknown"}
                
        except Exception as e:
            logger.error(f"Error getting date range: {e}")
            return {"min_date": "Unknown", "max_date": "Unknown"}
    
    def execute_sql_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            logger.info(f"Executing SQL: {query[:100]}...")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(query)
            
            if query.strip().upper().startswith('SELECT'):
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                results = [dict(zip(columns, row)) for row in rows]
                
                conn.close()
                logger.info(f"Query executed successfully, returned {len(results)} rows")
                return {
                    "success": True,
                    "data": results,
                    "row_count": len(results),
                    "columns": columns
                }
            else:
                conn.commit()
                conn.close()
                return {
                    "success": True,
                    "message": "Query executed successfully",
                    "affected_rows": cursor.rowcount
                }
                
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def generate_sql_query(self, state: WorkflowState) -> WorkflowState:
        """Generate SQL query from natural language"""
        query = state["user_query"]
        
        context = ""
        if self.knowledge_agent:
            context = self.knowledge_agent.get_context(query)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL query generator for a superstore database.
            
            Database Schema and Context:
            {context}
            
            Rules:
            1. Use only tables: sales_data, base_forecasts
            2. Return ONLY the SQL query without markdown, explanations, or formatting
            3. Use proper SQLite syntax
            4. Include appropriate WHERE, GROUP BY, ORDER BY, and LIMIT clauses
            5. For date filtering use: DATE('now', '-N days') or strftime functions
            6. Do NOT include ```sql or ``` markers
            7. Handle forecast accuracy queries with proper LEFT JOINs
            
            Example formats:
            - Simple: SELECT customer_name, SUM(sales) FROM sales_data GROUP BY customer_name ORDER BY SUM(sales) DESC LIMIT 10
            - Time series: SELECT strftime('%Y-%m', order_date) as month, SUM(sales) FROM sales_data GROUP BY month ORDER BY month
            - Forecast join: SELECT f.forecast_date, AVG(f.forecasted_sales) as forecast, AVG(COALESCE(s.sales, 0)) as actual FROM base_forecasts f LEFT JOIN sales_data s ON f.product_id = s.product_id AND f.forecast_date = s.order_date WHERE f.forecast_date >= DATE('now', '-30 days') GROUP BY f.forecast_date"""),
            ("human", "Convert this to SQL: {query}")
        ])
        
        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({"context": context, "query": query})
            
            sql_query = self._clean_sql_query(response.content)
            
            state["sql_query"] = sql_query
            state["current_agent"] = "sql_generator"
            
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            state["error"] = f"SQL generation error: {str(e)}"
        
        return state
    
    def _clean_sql_query(self, raw_sql: str) -> str:
        """Clean and format SQL query"""
        sql_query = raw_sql.strip()
        
        if "```" in sql_query:
            lines = sql_query.split('\n')
            sql_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or (not in_code_block and not line.strip().startswith('```')):
                    sql_lines.append(line)
            
            sql_query = '\n'.join(sql_lines).strip()
        
        sql_lines = []
        for line in sql_query.split('\n'):
            if '--' in line:
                line = line.split('--')[0]
            sql_lines.append(line)
        
        sql_query = '\n'.join(sql_lines).strip()
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        
        return sql_query
    
    async def execute_and_analyze(self, state: WorkflowState) -> WorkflowState:
        """Execute SQL and analyze results"""
        sql_query = state["sql_query"]
        
        results = self.execute_sql_query(sql_query)
        state["sql_results"] = results
        
        if not results["success"]:
            state["error"] = f"SQL execution failed: {results['error']}"
            return state
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze these SQL query results and provide business insights.
            
            Focus on:
            1. Key insights from the data
            2. Business implications  
            3. Notable patterns or trends
            4. Actionable recommendations
            
            Keep the analysis concise and business-focused."""),
            ("human", """Query: {query}
            Results: {results}
            
            Provide analysis:""")
        ])
        
        try:
            data_summary = {
                "row_count": results["row_count"],
                "columns": results["columns"],
                "sample_data": results["data"][:5] if results["data"] else []
            }
            
            chain = analysis_prompt | self.llm
            response = await chain.ainvoke({
                "query": sql_query, 
                "results": json.dumps(data_summary, indent=2)
            })
            
            state["final_response"] = response.content
            state["current_agent"] = "sql_analyzer"
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            state["error"] = f"Analysis error: {str(e)}"
        
        return state

# ==========================================
# WORKFLOW COORDINATOR
# ==========================================

class WorkflowCoordinator:
    """LangGraph-based workflow coordinator"""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-8b-8192",
            temperature=0.1
        )
        
        self.knowledge_agent = KnowledgeAgent(self.llm)
        self.sql_agent = SQLAgent(self.llm)
        self.sql_agent.set_knowledge_agent(self.knowledge_agent)
        self.forecasting_agent = AdvancedForecastingAgent(self.llm)

        self.workflow = self._build_workflow_enhanced()
    
    def _build_workflow_enhanced(self) -> StateGraph:
        """Enhanced workflow with forecasting"""
        
        workflow = StateGraph(WorkflowState)
        
        # Add all nodes including forecasting
        workflow.add_node("classify_query", self._classify_query_enhanced)
        workflow.add_node("knowledge_agent", self.knowledge_agent.process_knowledge_query)
        workflow.add_node("sql_generator", self.sql_agent.generate_sql_query)
        workflow.add_node("sql_executor", self.sql_agent.execute_and_analyze)
        workflow.add_node("mixed_processor", self._process_mixed_query)
        workflow.add_node("forecasting_agent", self.forecasting_agent.process_forecasting_query)
        
        # Set entry point
        workflow.set_entry_point("classify_query")
        
        # Add conditional routing with forecasting
        workflow.add_conditional_edges(
            "classify_query",
            self._route_query_enhanced,
            {
                "knowledge": "knowledge_agent",
                "sql": "sql_generator",
                "mixed": "mixed_processor",
                "forecasting": "forecasting_agent"
            }
        )
        
        # Connect existing edges
        workflow.add_edge("sql_generator", "sql_executor")
        
        # Add end nodes
        workflow.add_edge("knowledge_agent", END)
        workflow.add_edge("sql_executor", END)
        workflow.add_edge("mixed_processor", END)
        workflow.add_edge("forecasting_agent", END)
        
        return workflow.compile()
    
    async def _classify_query_enhanced(self, state: WorkflowState) -> WorkflowState:
        """Enhanced classification that includes forecasting"""
        query = state["user_query"].lower()
        
        # Quick keyword detection for forecasting
        forecast_keywords = [
            "forecast", "predict", "future", "projection", "next month", "next quarter", 
            "upcoming", "trend", "will sell", "expected sales", "predict sales",
            "sales forecast", "demand forecast", "future sales", "what will happen",
            "next week", "next year", "coming months", "predict demand"
        ]
        
        if any(keyword in query for keyword in forecast_keywords):
            state["query_type"] = "forecasting"
            state["current_agent"] = "classifier"
            return state
        
        # Use LLM for other classifications
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Classify this query as one of: "sql", "knowledge", "mixed", or "forecasting"
            
            - "sql": Data analysis, reports, trends, specific numbers from existing data
            - "knowledge": Database schema, business rules, column meanings
            - "mixed": Needs both data analysis AND schema/business context
            - "forecasting": Predictions, future sales, trends, "what will happen"
            
            Respond with only: sql, knowledge, mixed, or forecasting"""),
            ("human", "{query}")
        ])
        
        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({"query": state["user_query"]})
            classification = response.content.strip().lower()
            
            if classification in ["sql", "knowledge", "mixed", "forecasting"]:
                state["query_type"] = classification
            else:
                state["query_type"] = "mixed"  # Default fallback
            
            state["current_agent"] = "classifier"
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            state["query_type"] = "mixed"
            state["error"] = f"Classification error: {str(e)}"
        
        return state
    
    def _route_query_enhanced(self, state: WorkflowState) -> str:
        """Enhanced routing that includes forecasting"""
        return state["query_type"]
    
    async def _process_mixed_query(self, state: WorkflowState) -> WorkflowState:
        """Handle queries requiring both agents"""
        
        knowledge_state = await self.knowledge_agent.process_knowledge_query(state)
        
        sql_state = await self.sql_agent.generate_sql_query(knowledge_state)
        if not sql_state.get("error"):
            sql_state = await self.sql_agent.execute_and_analyze(sql_state)
        
        if sql_state.get("final_response") and knowledge_state.get("knowledge_context"):
            combined_response = f"""Business Context:
{knowledge_state['knowledge_context']}

Data Analysis:
{sql_state['final_response']}"""
            
            sql_state["final_response"] = combined_response
            sql_state["current_agent"] = "mixed"
        
        return sql_state
    
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process user query through the workflow"""
        
        initial_state = WorkflowState(
            messages=[HumanMessage(content=user_query)],
            user_query=user_query,
            query_type="",
            sql_query=None,
            sql_results=None,
            knowledge_context=None,
            final_response=None,
            current_agent="",
            error=None,
            forecast_data=None
        )
        
        try:
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Return results
            return {
                "success": True,
                "agent": final_state["current_agent"],
                "query_type": final_state["query_type"],
                "response": final_state.get("final_response"),
                "sql_query": final_state.get("sql_query"),
                "sql_results": final_state.get("sql_results"),
                "knowledge_context": final_state.get("knowledge_context"),
                "forecast_data": final_state.get("forecast_data"),
                "error": final_state.get("error")
            }
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# ==========================================
# MAIN APPLICATION
# ==========================================

async def main():
    """Main interactive application"""
    
    print("🚀 LangGraph Multi-Agent Workflow System")
    print("SQL Database Agent + Knowledge RAG Agent + Forecasting Agent")
    print("Powered by Groq LLM API & LangGraph")
    print("=" * 60)
    
    try:
        # Initialize workflow coordinator
        print("🔧 Initializing LangGraph workflow...")
        coordinator = WorkflowCoordinator()
        print("✅ Workflow initialized successfully!")
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n💡 Please ensure you have:")
        print("1. Added GROQ_API_KEY to your .env file")
        print("2. Get free API key from: https://console.groq.com/")
        return
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return
    
    print(f"\n💬 Interactive Mode")
    print("Ask questions about your superstore data!")
    print("\n📊 Example Queries:")
    print("Knowledge Agent:")
    print("• 'What columns are in the sales data table?'")
    print("• 'Explain the business rules for the superstore data'")
    print("\nSQL Agent:")
    print("• 'Show me top 5 customers by sales'")
    print("• 'Analyze sales performance by region'")
    print("• 'What are the monthly sales trends?'")
    print("\nForecasting Agent:")
    print("• 'Forecast sales for the next 30 days'")
    print("• 'Predict weekly sales for the West region'")
    print("• 'What will be the sales trend for next month?'")
    print("• 'Generate a 3-month sales forecast'")
    print("\nType 'quit' to exit")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n🔍 Your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            print("⏳ Processing through LangGraph workflow...")
            result = await coordinator.process_query(user_input)
            
            if result["success"]:
                print(f"\n🤖 Agent: {result['agent']} (Type: {result['query_type']})")
                
                if result.get('response'):
                    print(f"\n📝 Response:")
                    print(result['response'])
                
                if result.get('sql_query'):
                    print(f"\n🔍 Generated SQL:")
                    print(result['sql_query'])
                
                if result.get('sql_results') and result['sql_results'].get('success'):
                    data = result['sql_results'].get('data', [])
                    if data:
                        print(f"\n📊 Query Results ({len(data)} rows):")
                        for i, row in enumerate(data[:3]):
                            print(f"   {i+1}. {row}")
                        if len(data) > 3:
                            print(f"   ... and {len(data) - 3} more rows")
                
                if result.get('forecast_data'):
                    forecast = result['forecast_data']
                    print(f"\n📈 Forecast Summary:")
                    print(f"   Method: {forecast.get('method', 'Unknown')}")
                    print(f"   Aggregation: {forecast.get('aggregation', 'daily')}")
                    print(f"   Predictions: {len(forecast.get('predictions', []))} points")
                    
            else:
                print(f"❌ Error: {result['error']}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Thanks for using the LangGraph Multi-Agent System!")

if __name__ == "__main__":
    asyncio.run(main())