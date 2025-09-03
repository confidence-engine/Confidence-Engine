#!/bin/bash
# Dual Agent Startup Script
# Run both main agent and high-risk futures agent independently

echo "🚀 Starting Dual Agent System"
echo "================================"

# Function to check if process is running
is_running() {
    pgrep -f "$1" > /dev/null
}

# Function to start main agent
start_main_agent() {
    echo "🔄 Starting Main Agent (Enhanced Hybrid with Intelligent TP/SL)..."
    if is_running "hybrid_crypto_trader.py"; then
        echo "✅ Main agent already running"
    else
        # Set enhanced environment variables
        export TB_INTELLIGENT_CRYPTO_TPSL=1  # Enable intelligent TP/SL
        export TB_USE_ENHANCED_RISK=1
        export TB_USE_ADAPTIVE_STRATEGY=1
        export TB_USE_REGIME_DETECTION=1
        
        nohup bash scripts/start_hybrid_loop.sh > main_agent.log 2>&1 &
        echo "✅ Enhanced main agent started (PID: $!) with intelligent TP/SL"
    fi
}

# Function to start futures agent
start_futures_agent() {
    echo "🔄 Starting Futures Agent (High-Risk with Intelligent TP/SL)..."
    if is_running "high_risk_futures_agent.py"; then
        echo "✅ Futures agent already running"
    else
        # Load futures-specific config and enable intelligent TP/SL
        export $(grep -v '^#' futures_agent_config.env | xargs) 2>/dev/null || true
        export TB_INTELLIGENT_FUTURES_TPSL=1  # Enable intelligent TP/SL
        
        nohup python3 high_risk_futures_agent.py --continuous > futures_agent.log 2>&1 &
        echo "✅ Enhanced futures agent started (PID: $!) with intelligent TP/SL"
    fi
}

# Function to stop agents
stop_agents() {
    echo "🛑 Stopping agents..."

    # Stop main agent
    if is_running "hybrid_crypto_trader.py"; then
        pkill -f "hybrid_crypto_trader.py"
        echo "✅ Main agent stopped"
    else
        echo "ℹ️  Main agent not running"
    fi

    # Stop futures agent
    if is_running "high_risk_futures_agent.py"; then
        pkill -f "high_risk_futures_agent.py"
        echo "✅ Futures agent stopped"
    else
        echo "ℹ️  Futures agent not running"
    fi
}

# Function to check status
check_status() {
    echo "📊 Agent Status"
    echo "---------------"

    # Main agent status
    if is_running "hybrid_crypto_trader.py"; then
        echo "✅ Main Agent: RUNNING"
        # Show recent log entries
        if [ -f main_agent.log ]; then
            echo "   📝 Recent activity:"
            tail -3 main_agent.log | sed 's/^/      /'
        fi
    else
        echo "❌ Main Agent: STOPPED"
    fi

    # Futures agent status
    if is_running "high_risk_futures_agent.py"; then
        echo "✅ Futures Agent: RUNNING"
        # Show recent log entries
        if [ -f futures_agent.log ]; then
            echo "   📝 Recent activity:"
            tail -3 futures_agent.log | sed 's/^/      /'
        fi
    else
        echo "❌ Futures Agent: STOPPED"
    fi
}

# Function to show logs
show_logs() {
    echo "📋 Agent Logs"
    echo "============="

    case $1 in
        main)
            if [ -f main_agent.log ]; then
                echo "🔍 Main Agent Log:"
                tail -20 main_agent.log
            else
                echo "❌ Main agent log not found"
            fi
            ;;
        futures)
            if [ -f futures_agent.log ]; then
                echo "🔍 Futures Agent Log:"
                tail -20 futures_agent.log
            else
                echo "❌ Futures agent log not found"
            fi
            ;;
        *)
            echo "📋 Both Agent Logs:"
            echo ""
            if [ -f main_agent.log ]; then
                echo "🔍 Main Agent (Last 10 lines):"
                tail -10 main_agent.log
                echo ""
            fi
            if [ -f futures_agent.log ]; then
                echo "🔍 Futures Agent (Last 10 lines):"
                tail -10 futures_agent.log
            fi
            ;;
    esac
}

# Main script logic
case $1 in
    start)
        start_main_agent
        echo ""
        start_futures_agent
        echo ""
        echo "🎯 Both agents started!"
        echo "💡 Monitor with: ./dual_agent.sh status"
        ;;
    stop)
        stop_agents
        ;;
    restart)
        stop_agents
        echo ""
        start_main_agent
        echo ""
        start_futures_agent
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs $2
        ;;
    main)
        case $2 in
            start)
                start_main_agent
                ;;
            stop)
                if is_running "hybrid_crypto_trader.py"; then
                    pkill -f "hybrid_crypto_trader.py"
                    echo "✅ Main agent stopped"
                fi
                ;;
            logs)
                show_logs main
                ;;
            *)
                echo "Usage: ./dual_agent.sh main {start|stop|logs}"
                ;;
        esac
        ;;
    futures)
        case $2 in
            start)
                start_futures_agent
                ;;
            stop)
                if is_running "high_risk_futures_agent.py"; then
                    pkill -f "high_risk_futures_agent.py"
                    echo "✅ Futures agent stopped"
                fi
                ;;
            logs)
                show_logs futures
                ;;
            *)
                echo "Usage: ./dual_agent.sh futures {start|stop|logs}"
                ;;
        esac
        ;;
    *)
        echo "🚀 Dual Agent Management Script"
        echo "==============================="
        echo ""
        echo "🧠 ENHANCED WITH INTELLIGENT TP/SL SYSTEM"
        echo "📊 Crypto: Trade-quality based targets (Excellent: 12-20%, Good: 8-12%, Fair: 5-8%)"
        echo "🚀 Futures: Leverage-adjusted targets (Excellent: 15-25%, Good: 10-15%, Fair: 6-10%)"
        echo ""
        echo "Usage:"
        echo "  ./dual_agent.sh start          # Start both enhanced agents"
        echo "  ./dual_agent.sh stop           # Stop both agents"
        echo "  ./dual_agent.sh restart        # Restart both agents"
        echo "  ./dual_agent.sh status         # Check agent status"
        echo "  ./dual_agent.sh logs           # Show both agent logs"
        echo "  ./dual_agent.sh logs main      # Show main agent logs"
        echo "  ./dual_agent.sh logs futures   # Show futures agent logs"
        echo "  ./dual_agent.sh main start     # Start only main agent"
        echo "  ./dual_agent.sh main stop      # Stop only main agent"
        echo "  ./dual_agent.sh futures start  # Start only futures agent"
        echo "  ./dual_agent.sh futures stop   # Stop only futures agent"
        echo ""
        echo "Examples:"
        echo "  ./dual_agent.sh start          # Start everything with intelligent TP/SL"
        echo "  ./dual_agent.sh status         # Check what's running"
        echo "  ./dual_agent.sh logs futures   # Monitor futures agent"
        ;;
esac

echo ""
