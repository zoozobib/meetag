#!/bin/bash
set -e

# Default to release mode
BUILD_MODE="release"
TAURI_FLAG=""
TARGET_DIR="target/release/bundle/macos"

# Check for arguments
if [[ "$1" == "--debug" ]]; then
    echo "🐞 Debug mode enabled"
    BUILD_MODE="debug"
    TAURI_FLAG="--debug"
    TARGET_DIR="target/debug/bundle/macos"
fi

# Build the Tauri app
echo "🚀 Starting $BUILD_MODE build..."
cargo tauri build $TAURI_FLAG

# Define paths
APP_NAME="rec"
APP_BUNDLE="$TARGET_DIR/$APP_NAME.app"

# Check if build succeeded
if [ ! -d "$APP_BUNDLE" ]; then
    echo "❌ Build failed or app bundle not found at $APP_BUNDLE"
    exit 1
fi

# Run the fix script
echo "🔧 Applying bundle/rpath fixes..."
./fix_bundle.sh "$APP_BUNDLE"

echo "✅ $BUILD_MODE build and fix completed successfully!"
echo "📦 App bundle: $APP_BUNDLE"
if [[ "$BUILD_MODE" == "release" ]]; then
    DMG_PATH="$TARGET_DIR/../dmg/${APP_NAME}_${VERSION}_aarch64.dmg" # Assuming version and arch, better to find
    # More robust find:
    DMG_PATH=$(find "$TARGET_DIR/../dmg" -name "*.dmg" -maxdepth 1 | head -n 1)
    if [[ -n "$DMG_PATH" ]]; then
        echo "📀 DMG found at: $DMG_PATH"
        # open "$DMG_PATH" # Optional: auto-open
    else
        echo "⚠️ DMG not found in expected location."
    fi
elif [[ "$BUILD_MODE" == "debug" ]]; then
    # Debug builds often produce DMGs too depending on config, but verify path
    DMG_PATH=$(find "$TARGET_DIR/../dmg" -name "*.dmg" -maxdepth 1 | head -n 1)
    if [[ -n "$DMG_PATH" ]]; then
        echo "📀 DMG found at: $DMG_PATH"
        echo "🚀 Opening DMG for testing..."
        open "$DMG_PATH"
    else 
        echo "⚠️ DMG (if configured) should be in target/debug/bundle/dmg/"
    fi
fi
