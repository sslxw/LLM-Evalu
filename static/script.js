document.addEventListener("DOMContentLoaded", () => {
    console.log("Document loaded and script initialized.");
    const socket = io();

    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
    });

    socket.on('connect_error', (err) => {
        console.log('Connection Error:', err);
    });

    const form = document.getElementById("query-form");
    const queryInput = document.getElementById("query");

    const gpt35ResponseBlock = document.getElementById("gpt3.5-response").querySelector("p");
    const gpt4ResponseBlock = document.getElementById("gpt4-response").querySelector("p");
    const llama2ResponseBlock = document.getElementById("llama2-response").querySelector("p");
    const falconResponseBlock = document.getElementById("falcon-response").querySelector("p");
    const bestResponseBlock = document.getElementById("best-response").querySelector("p");

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const query = queryInput.value;
        console.log(`Emitting new query: ${query}`);
        socket.emit("new_query", { query: query });

        gpt35ResponseBlock.textContent = "Waiting for response...";
        gpt4ResponseBlock.textContent = "Waiting for response...";
        llama2ResponseBlock.textContent = "Waiting for response...";
        falconResponseBlock.textContent = "Waiting for response...";
        bestResponseBlock.textContent = "Waiting for evaluation...";
    });

    socket.on("response", (data) => {
        console.log(`Received response for ${data.model}: ${data.response}`);
        if (data.model === "gpt35") {
            gpt35ResponseBlock.textContent = data.response;
        } else if (data.model === "gpt4") {
            gpt4ResponseBlock.textContent = data.response;
        } else if (data.model === "llama2") {
            llama2ResponseBlock.textContent = data.response;
        } else if (data.model === "falcon") {
            falconResponseBlock.textContent = data.response;
        }
    });

    socket.on("best_response", (data) => {
        console.log(`Received best response: ${data.response}`);
        bestResponseBlock.textContent = data.response;
    });
});