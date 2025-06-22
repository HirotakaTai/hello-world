document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    
    // クライアントIDを生成（簡易的なもの）
    const clientId = 'client-' + Date.now() + Math.random().toString(36).substring(2, 9);
    
    // WebSocketの接続（HTTPSかどうかでwssかwsを使い分ける）
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${clientId}`;
    console.log('WebSocket接続先URL:', wsUrl);
    
    const ws = new WebSocket(wsUrl);
    
    // 接続確立時の処理
    ws.onopen = () => {
        console.log('WebSocket接続が確立されました');
        // 接続確立後にウェルカムメッセージを表示
        addBotMessage('こんにちは！どのようにお手伝いできますか？');
    };
    
    // メッセージ受信時の処理
    ws.onmessage = (event) => {
        const response = JSON.parse(event.data);
        
        // 応答を表示
        if (response.status === 'success') {
            // タイピングインジケータを削除
            removeTypingIndicator();
            addBotMessage(response.response);
        } else {
            // エラーの場合
            removeTypingIndicator();
            addBotMessage('申し訳ありませんが、エラーが発生しました。もう一度お試しください。');
            console.error('エラー:', response.response);
        }
    };
    
    // エラー発生時の処理
    ws.onerror = (error) => {
        console.error('WebSocket エラー詳細:', error);
        // オブジェクトの内容を詳しく確認
        for (let prop in error) {
            if (error.hasOwnProperty(prop)) {
                console.error(`${prop}: ${error[prop]}`);
            }
        }
        removeTypingIndicator();
        addBotMessage('サーバーとの接続中にエラーが発生しました。コンソールで詳細を確認してください。');
    };
    
    // 接続切断時の処理
    ws.onclose = (event) => {
        console.log('WebSocket 接続が切断されました - コード:', event.code, '理由:', event.reason);
        addBotMessage(`サーバーとの接続が切断されました（コード: ${event.code}）。ページを再読み込みしてください。`);
    };
    
    // 送信ボタンのクリックイベント
    sendButton.addEventListener('click', sendMessage);
    
    // Enterキーの押下イベント
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
    
    // メッセージ送信関数
    function sendMessage() {
        const message = userInput.value.trim();
        
        if (message) {
            // ユーザーメッセージをUIに追加
            addUserMessage(message);
            
            // WebSocketでメッセージを送信
            ws.send(JSON.stringify({ message }));
            
            // 入力フィールドをクリア
            userInput.value = '';
            
            // タイピングインジケータを表示
            showTypingIndicator();
        }
    }
    
    // ユーザーメッセージをUIに追加
    function addUserMessage(text) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', 'user-message');
        messageElement.textContent = text;
        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }
    
    // ボットメッセージをUIに追加
    function addBotMessage(text) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', 'bot-message');
        messageElement.textContent = text;
        chatMessages.appendChild(messageElement);
        scrollToBottom();
    }
    
    // タイピングインジケータを表示
    function showTypingIndicator() {
        const typingElement = document.createElement('div');
        typingElement.classList.add('typing-indicator');
        typingElement.id = 'typing-indicator';
        typingElement.innerHTML = `
            <span></span>
            <span></span>
            <span></span>
        `;
        chatMessages.appendChild(typingElement);
        scrollToBottom();
    }
    
    // タイピングインジケータを削除
    function removeTypingIndicator() {
        const typingElement = document.getElementById('typing-indicator');
        if (typingElement) {
            typingElement.remove();
        }
    }
    
    // チャット領域を下にスクロール
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
