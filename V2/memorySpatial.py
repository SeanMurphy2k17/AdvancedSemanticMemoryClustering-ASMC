from spatialValenceCompute import spatialValenceCompute
from datetime import datetime, timezone

class memorySpatial:
    def __init__(self):
        print(f"initialized {self.__class__.__name__}")
        self._svc = spatialValenceCompute()

    def buildMemoryObject(self, inputText: str, responseText: str,
                          prevPos: tuple = None,
                          linkedMemories: list = None,
                          metaDataTag: dict = None) -> dict:
        return {
            "inputText":       inputText,
            "inputPos":        self._svc.computeSpatialValence(inputText),
            "responseText":    responseText,
            "responsePos":     self._svc.computeSpatialValence(responseText),
            "contentWords":    self._svc.extractContentWords(inputText),
            "prevPos":         prevPos,
            "linkedMemories":  linkedMemories or [],
            "metaDataTag":     metaDataTag or {},
            "timeDate":        datetime.now(timezone.utc).isoformat(),
        }


if __name__ == "__main__":
    _ms = memorySpatial()
    _obj = _ms.buildMemoryObject(
        inputText    = "my back is absolutely killing me",
        responseText = "that sounds really painful, have you tried resting?",
        prevPos      = None,
        linkedMemories = [],
        metaDataTag  = {"source": "chat", "agent": "test"},
    )
    for k, v in _obj.items():
        print(f"  {k}: {v}")
