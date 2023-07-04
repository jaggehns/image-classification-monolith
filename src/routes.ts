import { Express, Request, Response } from "express";
const { spawn } = require("child_process");

function routes(app: Express) {
  app.get("/image-classification-analytics", (req: Request, res: Response) => {
    const pythonScript = spawn("python", ["classify_clothes.py"]);

    pythonScript.stdout.on("data", (data: any) => {
      console.log(`Python script stdout: ${data}`);
    });

    pythonScript.stderr.on("data", (data: any) => {
      console.error(`Python script stderr: ${data}`);
    });

    pythonScript.on("close", (code: any) => {
      console.log(`Python script process exited with code ${code}`);
      res.send("Python script execution completed.");
    });
  });

  app.get("/classify-image", (req: Request, res: Response) => {
    console.time("f1");
    const pythonScript = spawn("python", ["classify_image.py"]);

    let scriptOutput: string = "";

    pythonScript.stdout.on("data", (data: any) => {
      scriptOutput = data.toString();
      console.log(`Python script stdout: ${data}`);
    });

    pythonScript.stderr.on("data", (data: any) => {
      console.error(`Python script stderr: ${data}`);
    });

    pythonScript.on("close", (code: any) => {
      console.log(`Python script process exited with code ${code}`);
      const outputLines = scriptOutput.split("\n").filter(Boolean); // Split output on newline and remove empty lines
      const predictedLabel = outputLines[0].split(": ")[1]; // Extract predicted label
      console.log(predictedLabel);
      const predictedClothingType = outputLines[1].split(": ")[1]; // Extract predicted clothing type

      const responseObj = {
        output: {
          "Predicted Label": predictedLabel,
          "Predicted Clothing Type": predictedClothingType,
        },
      };
      res.json(responseObj); // Send the response as JSON
      console.timeEnd("f1");
    });
  });
}

export default routes;
