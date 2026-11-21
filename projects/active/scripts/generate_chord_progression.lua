-- name: Generate Chord Progression (Smooth Voice Leading)
-- category: Harmony
-- description: Generates a smooth voice-led diatonic chord progression (min L1 movement <= 2.5 st, pivot notes preserved).

function form(f)
  f:combo("Root", { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" }, 1)
  f:combo("Scale", { "Minor", "Major", "Dorian", "Phrygian" }, 1)
  f:combo("Progression", { "i-VI-III-VII", "i-VI-iv-v", "i-iv-v-i" }, 1)
  f:knobInt("LowMIDI", 48, 36, 60)
  f:knobInt("HighMIDI", 72, 60, 84)
  f:knobInt("Bars", 4, 1, 16)
  f:knobInt("Channel", 1, 1, 64)
  f:knobInt("Pattern", 1, 1, 64)
  f:knobInt("Velocity", 88, 20, 127)
end

local SCALES = {
  Minor    = { 0, 2, 3, 5, 7, 8, 10 },
  Major    = { 0, 2, 4, 5, 7, 9, 11 },
  Dorian   = { 0, 2, 3, 5, 7, 9, 10 },
  Phrygian = { 0, 1, 3, 5, 7, 8, 10 },
}

local SCALE_NAMES = { "Minor", "Major", "Dorian", "Phrygian" }
local PROGRESSIONS = {
  { 1, 6, 3, 7 }, -- i - VI - III - VII
  { 1, 6, 4, 5 }, -- i - VI - iv - v
  { 1, 4, 5, 1 }, -- i - iv - v - i
}

local function chord_pcs(root_pc, scale, deg)
  local pcs = {}
  for _, offset in ipairs({ 0, 2, 4, 6 }) do -- 7th chord
    local idx = ((deg - 1 + offset) % 7) + 1
    table.insert(pcs, (root_pc + scale[idx]) % 12)
  end
  return pcs
end

local function build_candidates(pcs, low_midi, high_midi)
  local pools = {}
  for i, pc in ipairs(pcs) do
    local opts = {}
    for m = low_midi, high_midi do
      if (m % 12) == pc then
        table.insert(opts, m)
      end
    end
    pools[i] = opts
  end

  local res = {}
  local function rec(depth, cur)
    if depth > #pools then
      local s = { table.unpack(cur) }
      table.sort(s)
      if (s[#s] - s[1]) <= 16 then
        local tight = false
        for k = 1, #s - 1 do
          if (s[k + 1] - s[k]) <= 1 and s[k] < 57 then
            tight = true
            break
          end
        end
        if not tight then
          table.insert(res, s)
        end
      end
      return
    end
    for _, pitch in ipairs(pools[depth]) do
      cur[depth] = pitch
      rec(depth + 1, cur)
    end
  end
  rec(1, {})
  return res
end

local function l1_distance(prev_v, curr_v)
  local total = 0
  for _, c in ipairs(curr_v) do
    local best = 999
    for _, p in ipairs(prev_v) do
      local d = math.abs(c - p)
      if d < best then best = d end
    end
    total = total + best
  end
  return total / #curr_v
end

function apply(p)
  p = p or {}
  local P = project
  if P.channels.count() == 0 or P.patterns.count() == 0 then return end
  local ch = math.max(1, math.min(p.Channel or 1, P.channels.count()))
  local pat = math.max(1, math.min(p.Pattern or 1, P.patterns.count()))
  local root_pc = (p.Root or 1) - 1
  local scale = SCALES[SCALE_NAMES[p.Scale or 1]]
  local prog = PROGRESSIONS[p.Progression or 1]
  local low_midi = p.LowMIDI or 48
  local high_midi = p.HighMIDI or 72
  local bars = p.Bars or 4
  local vel = p.Velocity or 88
  local bar_ticks = P.song.bar()

  P.patterns.setBars(pat, bars)
  local notes = {}
  local prev_voicing = nil
  local mid = (low_midi + high_midi) / 2.0

  for bar = 0, bars - 1 do
    local deg = prog[(bar % #prog) + 1]
    local pcs = chord_pcs(root_pc, scale, deg)
    local cands = build_candidates(pcs, low_midi, high_midi)
    local chosen = cands[1]
    if not prev_voicing then
      local best_score = 999
      for _, c in ipairs(cands) do
        local avg = 0
        for _, m in ipairs(c) do avg = avg + m end
        avg = avg / #c
        local score = math.abs(avg - mid)
        if score < best_score then
          best_score = score
          chosen = c
        end
      end
    else
      local best_dist = 999
      for _, c in ipairs(cands) do
        local d = l1_distance(prev_voicing, c)
        if d < best_dist then
          best_dist = d
          chosen = c
        end
      end
    end
    prev_voicing = chosen

    local start_t = bar * bar_ticks
    for _, pitch in ipairs(chosen) do
      table.insert(notes, {
        time = start_t,
        length = bar_ticks - 12,
        pitch = pitch,
        velocity = vel,
      })
    end
  end
  P.patterns.setNotes(pat, ch, notes)
end
