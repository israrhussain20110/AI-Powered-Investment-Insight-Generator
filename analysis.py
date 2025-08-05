# analysis.py
class DataAnalyzer:
    """Generate unique AI-driven contextual insights."""
    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata
    
    def generate_insights(self):
        """Create narrative insights based on hype, sentiment, and spikes."""
        insights = {}
        for ticker in self.data.keys():
            try:
                spikes = self.metadata.get(ticker, {}).get("spike_count", 0)
                hype = self.metadata.get(ticker, {}).get("hype_score", 0)
                sentiment = self.metadata.get(ticker, {}).get("sentiment", 0)
                insight = f"Hype Score: {hype:.1f}. "
                if spikes > 0:
                    insight += f"{spikes} volume spikes detected. "
                if sentiment > 0.5:
                    insight += "Positive X sentiment suggests upside potential. "
                elif sentiment < -0.5:
                    insight += "Negative X sentiment warns of risk. "
                insights[ticker] = insight
            except Exception as e:
                print(f"Error generating insight for {ticker}: {e}")
                insights[ticker] = "Insight unavailable."
        return insights