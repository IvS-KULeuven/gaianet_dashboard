from pathlib import Path
import logging
import time
import sqlite3
import numpy as np

logger = logging.getLogger(__name__)


class DataLoaderSQLite:

    def __init__(self, path_to_data: Path):
        self._conn = sqlite3.connect(path_to_data / 'consolidated.db')
        self.cursor = self._conn.cursor()

    def retrieve_data(self, sid: str):
        self.cursor.execute(
            "SELECT lc, xp, dmdt FROM data WHERE sid = ?", (sid,)
        )
        result = self.cursor.fetchone()
        lc = np.frombuffer(result[0])
        n = lc.shape[0]//3
        lc = lc.reshape(3, n)
        xp = np.frombuffer(result[1]).reshape(2, 2, 55)
        dmdt = np.frombuffer(result[2], dtype=np.float32).reshape(23, 5)
        return lc, xp, dmdt

    def __del__(self):
        self._conn.close()

    def profile_test(self):
        tinit = time.time()
        sids = [
            40023739377166080,
            52968800869454080,
            46618953718068992,
            46309131957199360,
            47034706552255488,
            48103564995766528,
            40330159524038912,
            37123055904340992,
            52092941077056000,
            39412960668016896,
            48200910431063680,
            52956023344987136
        ]
        for sid in sids:
            self.retrieve_data(sid)
        logger.info(f"Retrieving data: {time.time()-tinit:0.4f}")
