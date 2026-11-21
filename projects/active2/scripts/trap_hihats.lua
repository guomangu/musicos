-- name: Trap Fast Hi-Hats & Rolls
-- category: Rhythm
-- description: Generates fast dark trap hi-hat patterns (1/8, 1/16, 1/32, triplet rolls, velocity ramps, pitch drops) in C minor arrangement patterns.

local function add_roll(notes, start_tick, count, step_ticks, note_len, pitch_start, pitch_end, vel_start, vel_end, pan_start, pan_end)
  for i = 0, count - 1 do
    local frac = (count > 1) and (i / (count - 1)) or 0
    local p = math.floor(pitch_start + (pitch_end - pitch_start) * frac + 0.5)
    local v = math.floor(vel_start + (vel_end - vel_start) * frac + 0.5)
    local pan = math.floor(pan_start + (pan_end - pan_start) * frac + 0.5)
    table.insert(notes, {
      time = start_tick + i * step_ticks,
      length = note_len,
      pitch = math.max(36, math.min(84, p)),
      velocity = math.max(25, math.min(127, v)),
      pan = math.max(-64, math.min(64, pan)),
    })
  end
end

local function build_intro_hats(ppq)
  local notes = {}
  local bar = ppq * 4
  local eighth = math.floor(ppq / 2)
  local thirty_second = math.floor(ppq / 8)
  -- Sparse filtered-feel hats in bars 3 and 4
  for b = 2, 3 do
    for step = 0, 7 do
      if b == 3 and step == 7 then
        -- Pre-drop 1/32 roll
        add_roll(notes, b * bar + step * eighth, 4, thirty_second, thirty_second - 2, 60, 56, 52, 92, -14, 14)
      elseif (b == 2 and step % 2 == 0) or (b == 3 and step < 7) then
        local vel = 42 + b * 10 + (step % 2 == 0 and 10 or 0)
        table.insert(notes, {
          time = b * bar + step * eighth,
          length = thirty_second * 2,
          pitch = 60,
          velocity = vel,
          pan = (step % 2 == 0) and 4 or -4,
        })
      end
    end
  end
  return notes
end

local function build_drop_a_hats(ppq)
  local notes = {}
  local bar = ppq * 4
  local eighth = math.floor(ppq / 2)
  local sixteenth = math.floor(ppq / 4)
  local thirty_second = math.floor(ppq / 8)
  local triplet_16 = math.floor(ppq / 6)

  for b = 0, 3 do
    local b_start = b * bar
    for s = 0, 7 do
      local t = b_start + s * eighth
      if b == 0 and s == 3 then
        add_roll(notes, t, 4, thirty_second, thirty_second - 2, 60, 58, 90, 66, -12, 12)
      elseif b == 0 and s == 7 then
        add_roll(notes, t, 3, triplet_16, triplet_16 - 2, 58, 60, 68, 96, 10, -10)
      elseif b == 1 and s == 5 then
        add_roll(notes, t, 4, thirty_second, thirty_second - 2, 59, 56, 86, 62, -16, 16)
      elseif b == 2 and s == 3 then
        add_roll(notes, t, 4, thirty_second, thirty_second - 2, 60, 57, 92, 68, 14, -14)
      elseif b == 3 and s == 6 then
        add_roll(notes, t, 3, triplet_16, triplet_16 - 2, 60, 58, 84, 72, -10, 0)
      elseif b == 3 and s == 7 then
        add_roll(notes, t, 4, thirty_second, thirty_second - 2, 58, 55, 74, 102, 0, 18)
      else
        local vel = (s % 2 == 0) and 92 or 70
        if s == 0 then vel = 100 end
        local jitter = math.random(-4, 4)
        table.insert(notes, {
          time = t,
          length = sixteenth,
          pitch = 60,
          velocity = math.max(30, math.min(120, vel + jitter)),
          pan = (s % 2 == 0) and 6 or -6,
        })
        -- Extra 1/16 syncopated bounce on s == 1 of bars 1 & 3
        if (b == 1 or b == 3) and s == 1 then
          table.insert(notes, {
            time = t + sixteenth,
            length = thirty_second,
            pitch = 60,
            velocity = 64,
            pan = 10,
          })
        end
      end
    end
  end
  return notes
end

local function build_drop_b_hats(ppq)
  local notes = {}
  local bar = ppq * 4
  local eighth = math.floor(ppq / 2)
  local sixteenth = math.floor(ppq / 4)
  local thirty_second = math.floor(ppq / 8)
  local triplet_16 = math.floor(ppq / 6)
  local triplet_32 = math.floor(ppq / 12)

  for b = 0, 3 do
    local b_start = b * bar
    for s = 0, 7 do
      local t = b_start + s * eighth
      if b == 0 and s == 3 then
        add_roll(notes, t, 4, thirty_second, thirty_second - 2, 60, 57, 92, 66, -16, 16)
      elseif b == 1 and s == 6 then
        add_roll(notes, t, 3, triplet_16, triplet_16 - 2, 60, 59, 86, 72, 12, -4)
      elseif b == 1 and s == 7 then
        add_roll(notes, t, 6, triplet_32, triplet_32 - 1, 59, 55, 70, 98, -22, 22)
      elseif b == 2 and s == 5 then
        add_roll(notes, t, 4, thirty_second, thirty_second - 2, 58, 56, 88, 64, 16, -16)
      elseif b == 3 and s == 3 then
        add_roll(notes, t, 3, triplet_16, triplet_16 - 2, 60, 58, 84, 70, -12, 12)
      elseif b == 3 and s == 7 then
        add_roll(notes, t, 6, triplet_32, triplet_32 - 1, 60, 54, 72, 106, -24, 24)
      else
        local vel = (s % 2 == 0) and 94 or 72
        if s == 0 then vel = 102 end
        table.insert(notes, {
          time = t,
          length = sixteenth,
          pitch = 60,
          velocity = math.max(30, math.min(120, vel + math.random(-4, 4))),
          pan = (s % 2 == 0) and 6 or -6,
        })
        if s == 1 or (b % 2 == 1 and s == 4) then
          table.insert(notes, {
            time = t + sixteenth,
            length = thirty_second,
            pitch = 60,
            velocity = 62,
            pan = -10,
          })
        end
      end
    end
  end
  return notes
end

local function build_verse_a_hats(ppq)
  local notes = {}
  local bar = ppq * 4
  local eighth = math.floor(ppq / 2)
  local sixteenth = math.floor(ppq / 4)
  local triplet_16 = math.floor(ppq / 6)

  for b = 0, 3 do
    local b_start = b * bar
    for s = 0, 7 do
      local t = b_start + s * eighth
      -- Leave beat 3 (s == 4) open on bars 0 and 2 for clap pocket
      if (b == 0 or b == 2) and s == 4 then
        -- subtle pause
      elseif (b == 1 or b == 3) and s == 7 then
        add_roll(notes, t, 3, triplet_16, triplet_16 - 2, 60, 57, 78, 58, -12, 12)
      else
        local vel = (s % 2 == 0) and 82 or 60
        table.insert(notes, {
          time = t,
          length = sixteenth,
          pitch = 60,
          velocity = vel + math.random(-4, 4),
          pan = (s % 2 == 0) and 5 or -5,
        })
      end
    end
  end
  return notes
end

local function build_verse_b_hats(ppq)
  local notes = {}
  local bar = ppq * 4
  local eighth = math.floor(ppq / 2)
  local sixteenth = math.floor(ppq / 4)
  local thirty_second = math.floor(ppq / 8)

  for b = 0, 3 do
    local b_start = b * bar
    for s = 0, 7 do
      local t = b_start + s * eighth
      if b == 3 and s == 6 then
        add_roll(notes, t, 2, sixteenth, sixteenth - 2, 60, 59, 76, 86, -8, 8)
      elseif b == 3 and s == 7 then
        add_roll(notes, t, 4, thirty_second, thirty_second - 2, 59, 55, 86, 108, -18, 18)
      elseif b == 1 and s == 3 then
        add_roll(notes, t, 4, thirty_second, thirty_second - 2, 60, 58, 86, 62, 12, -12)
      else
        local vel = (s % 2 == 0) and 88 or 66
        table.insert(notes, {
          time = t,
          length = sixteenth,
          pitch = 60,
          velocity = vel + math.random(-4, 4),
          pan = (s % 2 == 0) and 6 or -6,
        })
      end
    end
  end
  return notes
end

function apply()
  local P = project
  local ppq = P.song.ppq()
  local ch = P.channels.find("Hi-Hat Closed")
  if not ch then
    error("Channel 'Hi-Hat Closed' not found")
  end

  local builders = {
    ["Intro Drums"] = build_intro_hats,
    ["Drop A"]      = build_drop_a_hats,
    ["Drop B"]      = build_drop_b_hats,
    ["Verse A"]     = build_verse_a_hats,
    ["Verse B"]     = build_verse_b_hats,
  }

  local updated = 0
  for pat_name, fn in pairs(builders) do
    local pat = P.patterns.find(pat_name)
    if pat then
      local list = fn(ppq)
      P.patterns.setNotes(pat, ch, list)
      log(string.format("Generated %d hi-hat notes in pattern '%s' (#%d)", #list, pat_name, pat))
      updated = updated + 1
    end
  end
  log("Trap hi-hats updated across " .. tostring(updated) .. " patterns.")
end
