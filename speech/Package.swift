// swift-tools-version: 5.10
// speechd: the ears and voice of the app. Argmax speech-to-text + text-to-speech, on device.
// One-time setup before `swift build` (already done on this Mac):
//   swift package-registry set --global --scope argmaxinc https://api.argmaxinc.com/v1/sdk
//   swift package-registry login https://api.argmaxinc.com/v1/sdk/login --token <axst_ token>
import PackageDescription

let package = Package(
    name: "speechd",
    platforms: [.macOS("15.0")],
    dependencies: [
        // Argmax Pro SDK (registry). It re-exports the open-source WhisperKit, TTSKit and SpeakerKit.
        .package(id: "argmaxinc.argmax-sdk-swift", from: "3.1.8"),
    ],
    targets: [
        .executableTarget(
            name: "speechd",
            dependencies: [.product(name: "Argmax", package: "argmaxinc.argmax-sdk-swift")]
        ),
    ]
)
