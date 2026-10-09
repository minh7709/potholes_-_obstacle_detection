plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Stamped at configure time, so it changes on every build. The HUD prints it: when a
// report still shows an old layout, this says whether the new APK was ever installed
// or the screenshot simply predates it.
val buildStamp: String = System.currentTimeMillis().toString()

android {
    namespace = "com.example.fcw"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.fcw"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        buildConfigField("String", "BUILD_STAMP", "\"$buildStamp\"")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    // Without this the debug APK bundles every ABI plus OpenCV and lands around
    // 210 MB. Splitting keeps a single-ABI APK installable for manual testing.
    // ONNX Runtime and OpenCV are both large native libraries, so this matters.
    splits {
        abi {
            isEnable = true
            reset()
            include("arm64-v8a", "armeabi-v7a")
            isUniversalApk = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

dependencies {
    val cameraxVersion = "1.4.1"

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-compose:1.9.3")

    // Compose
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3:1.3.1")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")

    // CameraX
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // ONNX Runtime Mobile. NNAPI gives the best chance of running the depth
    // model at a usable rate; CPU fallback keeps it correct if unavailable.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")

    // Coroutines for background inference
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    // OpenCV Android, used only for optical flow in the ego-motion stage
    implementation("org.opencv:opencv:4.10.0")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
}