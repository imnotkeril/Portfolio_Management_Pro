"""Fetch Fama-French factor data from Kenneth French Data Library."""

import logging
from datetime import date
from typing import Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

logger = logging.getLogger(__name__)

# Kenneth French Data Library URLs
FF_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF_3F_URL = FF_BASE_URL + "F-F_Research_Data_Factors_daily.CSV"
FF_5F_URL = FF_BASE_URL + "F-F_Research_Data_5_Factors_2x3_daily.CSV"
FF_MOM_URL = FF_BASE_URL + "F-F_Momentum_Factor_daily.CSV"


def fetch_fama_french_3factor(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch Fama-French 3-Factor model data.
    
    Factors: Mkt-RF (Market minus Risk-Free), SMB (Small minus Big), 
    HML (High minus Low book-to-market)
    
    Args:
        start_date: Optional start date
        end_date: Optional end date
        
    Returns:
        DataFrame with columns: Date, Mkt-RF, SMB, HML, RF (risk-free rate)
        Returns None if fetch fails
    """
    try:
        logger.info("Fetching Fama-French 3-Factor data...")
        
        # Read CSV from URL
        response = urlopen(FF_3F_URL, timeout=30)
        content = response.read().decode('utf-8')
        
        # Parse CSV (skip header lines)
        lines = content.strip().split('\n')
        
        # Find data start (skip header)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('19') or line.strip().startswith('20'):
                data_start = i
                break
        
        # Parse data
        data_rows = []
        for line in lines[data_start:]:
            if not line.strip() or line.strip().startswith('Copyright'):
                break
            
            parts = line.strip().split(',')
            if len(parts) >= 5:
                try:
                    date_str = parts[0].strip()
                    # Handle different date formats (YYYYMMDD or YYYY-MM-DD)
                    if len(date_str) == 8:
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                    else:
                        # Try parsing as ISO format
                        dt = pd.to_datetime(date_str).date()
                        year = dt.year
                        month = dt.month
                        day = dt.day
                    dt = date(year, month, day)
                    
                    mkt_rf = float(parts[1]) / 100  # Convert to decimal
                    smb = float(parts[2]) / 100
                    hml = float(parts[3]) / 100
                    rf = float(parts[4]) / 100
                    
                    data_rows.append({
                        'Date': dt,
                        'Mkt-RF': mkt_rf,
                        'SMB': smb,
                        'HML': hml,
                        'RF': rf,
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line: {line[:50]}... Error: {e}")
                    continue
        
        if not data_rows:
            logger.warning("No data rows parsed from Fama-French file")
            return None
        
        df = pd.DataFrame(data_rows)
        df.set_index('Date', inplace=True)
        
        # Filter by date range if provided
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        logger.info(
            f"Fetched {len(df)} days of Fama-French 3-Factor data "
            f"({df.index.min()} to {df.index.max()})"
        )
        
        return df
        
    except (HTTPError, URLError) as e:
        logger.error(f"Failed to fetch Fama-French data: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing Fama-French data: {e}", exc_info=True)
        return None


def fetch_fama_french_momentum(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Optional[pd.Series]:
    """
    Fetch Fama-French Momentum factor data.
    
    Args:
        start_date: Optional start date
        end_date: Optional end date
        
    Returns:
        Series with Momentum factor returns, indexed by Date
        Returns None if fetch fails
    """
    try:
        logger.info("Fetching Fama-French Momentum factor data...")
        
        response = urlopen(FF_MOM_URL, timeout=30)
        content = response.read().decode('utf-8')
        
        lines = content.strip().split('\n')
        
        # Find data start
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('19') or line.strip().startswith('20'):
                data_start = i
                break
        
        data_rows = []
        for line in lines[data_start:]:
            if not line.strip() or line.strip().startswith('Copyright'):
                break
            
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    date_str = parts[0].strip()
                    # Handle different date formats
                    if len(date_str) == 8:
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                    else:
                        dt_temp = pd.to_datetime(date_str).date()
                        year = dt_temp.year
                        month = dt_temp.month
                        day = dt_temp.day
                    dt = date(year, month, day)
                    
                    mom = float(parts[1]) / 100  # Convert to decimal
                    
                    data_rows.append({
                        'Date': dt,
                        'MOM': mom,
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line: {line[:50]}... Error: {e}")
                    continue
        
        if not data_rows:
            logger.warning("No data rows parsed from Momentum file")
            return None
        
        df = pd.DataFrame(data_rows)
        df.set_index('Date', inplace=True)
        
        # Filter by date range
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        logger.info(
            f"Fetched {len(df)} days of Momentum factor data "
            f"({df.index.min()} to {df.index.max()})"
        )
        
        return df['MOM']
        
    except (HTTPError, URLError) as e:
        logger.error(f"Failed to fetch Momentum factor data: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing Momentum data: {e}", exc_info=True)
        return None


def get_fama_french_factors(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    include_momentum: bool = True,
) -> Optional[Dict[str, pd.Series]]:
    """
    Get all Fama-French factors as a dictionary of Series.
    
    Args:
        start_date: Optional start date
        end_date: Optional end date
        include_momentum: Whether to include momentum factor
        
    Returns:
        Dictionary with factor names as keys and Series as values:
        {
            'Market (Mkt-RF)': Series,
            'Size (SMB)': Series,
            'Value (HML)': Series,
            'Momentum (MOM)': Series (if include_momentum=True),
        }
        Returns None if fetch fails
    """
    # Fetch 3-factor data
    ff3 = fetch_fama_french_3factor(start_date, end_date)
    if ff3 is None or ff3.empty:
        logger.warning("Failed to fetch Fama-French 3-Factor data")
        return None
    
    factors = {
        'Market (Mkt-RF)': ff3['Mkt-RF'],
        'Size (SMB)': ff3['SMB'],
        'Value (HML)': ff3['HML'],
    }
    
    # Add momentum if requested
    if include_momentum:
        mom = fetch_fama_french_momentum(start_date, end_date)
        if mom is not None and not mom.empty:
            factors['Momentum (MOM)'] = mom
        else:
            logger.warning("Failed to fetch Momentum factor")
    
    return factors

