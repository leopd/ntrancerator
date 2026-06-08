//! Spectrogram history ring/cursor logic (spec §8).
//!
//! The history texture is a fixed-width ring: each new column is written at the
//! current cursor `x`, then the cursor advances modulo the width. The fragment
//! shader uses the cursor to offset UVs so the newest column sits at the right
//! edge and history scrolls left. No full-texture shifting ever happens.
//!
//! This struct holds only the cursor arithmetic so the wraparound and the
//! shader UV offset are testable without a GPU.

/// Write cursor over a fixed-width history texture.
#[derive(Copy, Clone, Debug)]
pub struct HistoryCursor {
    width: u32,
    x: u32,
}

impl HistoryCursor {
    /// Create a cursor for a texture `width` columns wide.
    ///
    /// # Panics
    /// Panics if `width == 0`.
    pub fn new(width: u32) -> Self {
        assert!(width > 0, "history width must be > 0");
        Self { width, x: 0 }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    /// Current write position (the texture column the next column goes into).
    pub fn position(&self) -> u32 {
        self.x
    }

    /// Return the column index to write the next column into, then advance the
    /// cursor by one (wrapping at `width`).
    pub fn advance(&mut self) -> u32 {
        let pos = self.x;
        self.x = (self.x + 1) % self.width;
        pos
    }

    /// Horizontal UV offset for the shader so the newest column is at the right
    /// edge: the just-written column is at `x-1 (mod width)`, and that should map
    /// to UV `1.0`. Returned in `[0, 1)`.
    pub fn uv_offset(&self) -> f32 {
        self.x as f32 / self.width as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_wraps_around() {
        let mut c = HistoryCursor::new(4);
        assert_eq!(c.advance(), 0);
        assert_eq!(c.advance(), 1);
        assert_eq!(c.advance(), 2);
        assert_eq!(c.advance(), 3);
        // Wraps back to the start without overflow.
        assert_eq!(c.advance(), 0);
        assert_eq!(c.position(), 1);
    }

    #[test]
    fn uv_offset_tracks_cursor() {
        let mut c = HistoryCursor::new(8);
        assert_eq!(c.uv_offset(), 0.0);
        c.advance();
        assert_eq!(c.uv_offset(), 1.0 / 8.0);
    }

    #[test]
    fn width_one_always_writes_column_zero() {
        let mut c = HistoryCursor::new(1);
        for _ in 0..5 {
            assert_eq!(c.advance(), 0);
        }
    }

    #[test]
    #[should_panic]
    fn zero_width_panics() {
        HistoryCursor::new(0);
    }
}
