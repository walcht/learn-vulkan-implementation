#version 450
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec3 a_color;

layout(location = 0) out vec3 fragColor;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  fragColor = colors[gl_VertexIndex];
}
