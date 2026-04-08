import os
import json
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///portfolio.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Holding(Base):
    __tablename__ = "holdings"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    ticker = Column(String, default="")
    side = Column(String, nullable=False)
    category = Column(String, nullable=False)
    currency = Column(String, default="USD")
    quantity = Column(Float, default=1.0)
    cost_basis = Column(Float, default=0.0)
    current_value = Column(Float, default=0.0)
    date_added = Column(String)
    notes = Column(String, default="")

class Snapshot(Base):
    __tablename__ = "snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(String, nullable=False)
    net_worth = Column(Float, default=0.0)
    total_assets = Column(Float, default=0.0)
    total_liabilities = Column(Float, default=0.0)

Base.metadata.create_all(bind=engine)

def migrate_from_json():
    """Migrate data from legacy JSON to SQLite if the database is empty."""
    session = SessionLocal()
    
    # Check if empty
    if session.query(Holding).count() == 0 and os.path.exists("portfolio_data.json"):
        try:
            with open("portfolio_data.json") as f:
                data = json.load(f)
            for item in data:
                h = Holding(
                    id=item.get("id"),
                    name=item.get("Name"),
                    ticker=item.get("Ticker", ""),
                    side=item.get("Side"),
                    category=item.get("Category"),
                    currency=item.get("Currency", "USD"),
                    quantity=float(item.get("Quantity", 1.0)),
                    cost_basis=float(item.get("Cost Basis", 0.0)),
                    current_value=float(item.get("Current Value", 0.0)),
                    date_added=item.get("Date Added", ""),
                    notes=item.get("Notes", "")
                )
                session.add(h)
            session.commit()
            print("Successfully migrated holdings to SQLite.")
        except Exception as e:
            print("Migration of holdings failed:", e)
            session.rollback()

    if session.query(Snapshot).count() == 0 and os.path.exists("snapshots.json"):
        try:
            with open("snapshots.json") as f:
                snaps = json.load(f)
            for s in snaps:
                snap = Snapshot(
                    date=s.get("date"),
                    net_worth=float(s.get("net_worth", 0)),
                    total_assets=float(s.get("total_assets", 0)),
                    total_liabilities=float(s.get("total_liabilities", 0))
                )
                session.add(snap)
            session.commit()
            print("Successfully migrated snapshots to SQLite.")
        except Exception as e:
            print("Migration of snapshots failed:", e)
            session.rollback()
            
    session.close()

# Run migration on load
migrate_from_json()

def get_holdings_df():
    """Load holdings into the legacy pandas format."""
    df = pd.read_sql("SELECT * FROM holdings", con=engine)
    if not df.empty:
        # Standardize column names back to original for compatibility
        rename_map = {
            "name": "Name", "ticker": "Ticker", "side": "Side", 
            "category": "Category", "currency": "Currency", 
            "quantity": "Quantity", "cost_basis": "Cost Basis", 
            "current_value": "Current Value", "date_added": "Date Added", 
            "notes": "Notes"
        }
        df.rename(columns=rename_map, inplace=True)
    return df

def save_holdings_df(df: pd.DataFrame):
    """Save the pandas dataframe to the holdings table."""
    if df.empty:
        with engine.begin() as conn:
            conn.execute(Holding.__table__.delete())
        return
        
    df_sql = df.copy()
    rename_map = {
        "Name": "name", "Ticker": "ticker", "Side": "side", 
        "Category": "category", "Currency": "currency", 
        "Quantity": "quantity", "Cost Basis": "cost_basis", 
        "Current Value": "current_value", "Date Added": "date_added", 
        "Notes": "notes"
    }
    df_sql.rename(columns=rename_map, inplace=True)
    df_sql.to_sql("holdings", con=engine, if_exists="replace", index=False)
    
def get_snapshots_df():
    df = pd.read_sql("SELECT * FROM snapshots ORDER BY date ASC", con=engine)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df

def save_snapshot(date_str, net_worth, total_assets, total_liabilities):
    session = SessionLocal()
    snap = Snapshot(date=date_str, net_worth=net_worth, total_assets=total_assets, total_liabilities=total_liabilities)
    session.add(snap)
    session.commit()
    session.close()
