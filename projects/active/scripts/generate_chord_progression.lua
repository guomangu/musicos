-- name: Generate Chord Progression
-- category: Harmony
-- description: Generates a diatonic chord progression in the selected channel and pattern.

function form(f)
  f:combo("Root", { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" }, 1)
  f:combo("Scale", { "Minor", "Major", "Dorian", "Phrygian" }, 1)
  f:knobInt("Octave", 4, 2, 6)
  f:knobInt("Bars", 4, 1, 16)
  f:knobInt("Channel", 1, 1, 64)
  f:knobInt("Pattern", 1, 1, 64)
  f:knobInt("Velocity", 90, 20, 127)
end

local SCALES = {
  Minor    = { 0, 2, 3, 5, 7, 8, 10 },
  Major    = { 0, 2, 4, 5, 7, 9, 11 },
  Dorian   = { 0, 2, 3, 5, 7, 9, 10 },
  Phrygian = { 0, 1, 3, 5, 7, 8, 10 },
}

local SCALE_NAMES = { "Minor", "Major", "Dorian", "Phrygian" }
local PROGRESSION = { 1, 6, 3, 7 } -- i - VI - III - VII

function apply(p)
  p = p or {}
  local P = project
  if P.channels.count() == 0 or P.patterns.count() == 0 then return end
  local ch = math.max(1, math.min(p.Channel or 1, P.channels.count()))
  local pat = math.max(1, math.min(p.Pattern or 1, P.patterns.count()))
  local oct = p.Octave or 4
  local root = p.Root or 1
  local scale_idx = p.Scale or 1
  local bars = p.Bars or 4
  local vel = p.Velocity or 90
  local root_midi = 12 * (oct + 1) + (root - 1)
  local scale = SCALES[SCALE_NAMES[scale_idx]]
  local bar_ticks = P.song.bar()

  P.patterns.setBars(pat, bars)
  local notes = {}
  for bar = 0, bars - 1 do
    local deg = PROGRESSION[(bar % #PROGRESSION) + 1]
    local start_t = bar * bar_ticks
    for _, offset in ipairs({ 0, 2, 4 }) do
      local idx = ((deg - 1 + offset) % 7) + 1
      local oct_shift = math.floor((deg - 1 + offset) / 7) * 12
      table.insert(notes, {
        time = start_t,
        length = bar_ticks - 12,
        pitch = root_midi + scale[idx] + oct_shift,
        velocity = vel,
      })
    end
  end
  P.patterns.setNotes(pat, ch, notes)
end
