"""
V2X Gateway — Message Ingest, PKI Validation, and Broadcasting.
Acts as the entry point for all C-V2X (PC5 & 5G Uu) messages from vehicles/VRUs/RSUs.
"""
import time
import json
from core.v2x_models import BSMMessage, VRUMessage, RSUDetection


class V2XGateway:
    """
    5G MEC Gateway that receives raw V2X packets, validates PKI signatures,
    and forwards them to the Sensor Fusion layer.
    """

    VALID_CERTS = {"VALID_PKI_CERT_ECC_256", "EMERGENCY_CERT_ECC_384"}

    def __init__(self):
        self.packet_log: list = []           # Mobility Data Lake telemetry log
        self.rejected_count: int = 0
        self.total_ingested: int = 0

    def ingest_bsm(self, bsm: BSMMessage) -> dict | None:
        """
        Validate and ingest a Basic Safety Message (BSM) from a connected vehicle.
        Returns the message dict if valid, None if rejected.
        """
        self.total_ingested += 1

        if bsm.pki_signature not in self.VALID_CERTS:
            self.rejected_count += 1
            return None

        msg = bsm.model_dump()
        msg["source"] = "BSM"
        self.packet_log.append(msg)
        if len(self.packet_log) > 500:   # ring-buffer
            self.packet_log.pop(0)
        return msg

    def ingest_vru(self, vru: VRUMessage) -> dict:
        """
        Ingest Vulnerable Road User telemetry (phone/wearable app).
        VRU messages are not PKI-validated in this prototype (simplified).
        """
        self.total_ingested += 1
        msg = vru.model_dump()
        msg["source"] = "VRU"
        self.packet_log.append(msg)
        if len(self.packet_log) > 500:
            self.packet_log.pop(0)
        return msg

    def ingest_rsu(self, det: RSUDetection) -> dict:
        """Ingest RSU radar / camera detection."""
        self.total_ingested += 1
        msg = det.model_dump()
        msg["source"] = "RSU"
        self.packet_log.append(msg)
        if len(self.packet_log) > 500:
            self.packet_log.pop(0)
        return msg

    def get_recent_packets(self, n: int = 10) -> list:
        """Return the n most recent packets for the live dashboard feed."""
        return self.packet_log[-n:]

    def stats(self) -> dict:
        return {
            "total_ingested": self.total_ingested,
            "rejected_packets": self.rejected_count,
            "packet_buffer_size": len(self.packet_log),
        }
