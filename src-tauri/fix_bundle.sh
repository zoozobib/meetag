#!/bin/bash
# fix_bundle.sh
# Fixes library paths in the built macOS bundle

APP_BUNDLE="$1"
if [ -z "$APP_BUNDLE" ]; then
    echo "Usage: fix_bundle.sh /path/to/YourApp.app"
    exit 1
fi

APP_NAME=$(basename "$APP_BUNDLE" .app)
BINARY="$APP_BUNDLE/Contents/MacOS/$APP_NAME"
FRAMEWORKS_DIR="$APP_BUNDLE/Contents/Frameworks"
RESOURCES_DIR="$APP_BUNDLE/Contents/Resources"

echo "🔧 Fixing bundle: $APP_BUNDLE"

# 1. Create Frameworks directory if missing
if [ ! -d "$FRAMEWORKS_DIR" ]; then
    mkdir -p "$FRAMEWORKS_DIR"
fi

# 2. Move ONNX Runtime library from Resources to Frameworks (where it belongs)
# We initially bundled it into Resources via tauri.conf.json
LIB_NAME="libonnxruntime.1.17.1.dylib"
if [ -f "$RESOURCES_DIR/$LIB_NAME" ]; then
    echo "📦 Moving $LIB_NAME to Frameworks..."
    mv "$RESOURCES_DIR/$LIB_NAME" "$FRAMEWORKS_DIR/"
else
    echo "⚠️ Warning: $LIB_NAME not found in Resources. Checking Frameworks..."
fi

# 3. Fix the library's own ID/install_name
# Only if it exists in Frameworks
if [ -f "$FRAMEWORKS_DIR/$LIB_NAME" ]; then
    echo "🏷️ Fixing install_name for $LIB_NAME..."
    install_name_tool -id "@rpath/$LIB_NAME" "$FRAMEWORKS_DIR/$LIB_NAME"
    
    # Sign the library (ad-hoc signing is better than nothing locally)
    codesign --force --sign - "$FRAMEWORKS_DIR/$LIB_NAME"
fi

# 4. Add @executable_path/../Frameworks to the main binary's rpath
echo "🔗 Adding rpath to $BINARY..."
install_name_tool -add_rpath "@executable_path/../Frameworks" "$BINARY" 2>/dev/null || echo "   (rpath might already exist)"

echo "✅ Bundle fix complete."
