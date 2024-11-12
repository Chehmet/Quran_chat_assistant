import unittest
from fastapi.testclient import TestClient
import webpage.api as api

# Assuming 'app' is the FastAPI app instance
app = api.app
client = TestClient(app)

class TestQuranAdvice(unittest.TestCase):
    
    def test_get_advice(self):
        response = client.post("/get_advice", json={"question": "What does the Quran say about patience?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("advice", response.json())

if __name__ == '__main__':
    unittest.main()
