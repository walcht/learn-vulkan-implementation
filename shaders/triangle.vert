#version 450

layout(binding = 0) uniform MVP {
  mat4 model;
  mat4 view;
  mat4 proj;
} u_mvp;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
layout(location = 2) in vec2 a_tex_coord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
  gl_Position = u_mvp.proj * u_mvp.view * u_mvp.model * vec4(a_position, 1.0f);
  fragColor = a_color;
  fragTexCoord = a_tex_coord;
}
