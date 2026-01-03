use webrtc_vad::Vad;

// VAD wrapper to allow sending to audio thread
pub struct SendVad(pub Vad);
unsafe impl Send for SendVad {}
unsafe impl Sync for SendVad {}
