<template>
  <div class="container mx-auto p-4">
    <h2 class="text-lg font-bold mb-4">Chuyển đổi Google Sheet sang dữ liệu LabelMe</h2>

    <div class="mb-4">
      <label class="block font-semibold mb-1">Google Sheet ID:</label>
      <input v-model="sheetId" type="text" class="border p-2 w-full" placeholder="Nhập Google Sheet ID">
    </div>

    <div class="mb-4">
      <label class="block font-semibold mb-1">Worksheet Name:</label>
      <input v-model="worksheetName" type="text" class="border p-2 w-full" placeholder="Nhập Worksheet Name">
    </div>

    <button @click="processImages" class="bg-blue-500 text-white p-2 rounded" :disabled="loading">
      {{ loading ? "Đang xử lý..." : "Bắt đầu" }}
    </button>

    <p v-if="message" class="mt-4 font-semibold text-green-600">{{ message }}</p>
    <p v-if="errorMessage" class="mt-4 font-semibold text-red-600">{{ errorMessage }}</p>
  </div>
</template>

<script>
export default {
  data() {
    return {
      sheetId: "",
      worksheetName: "",
      loading: false,
      message: "",
      errorMessage: ""
    };
  },
  methods: {
    async processImages() {
      if (!this.sheetId || !this.worksheetName) {
        this.errorMessage = "Vui lòng nhập đầy đủ Google Sheet ID và Worksheet Name!";
        return;
      }

      this.loading = true;
      this.message = "";
      this.errorMessage = "";

      try {
        const response = await fetch("http://localhost:8080/process/", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: `sheet_id=${this.sheetId}&worksheet_name=${this.worksheetName}`
        });

        if (!response.ok) {
          throw new Error("Lỗi khi xử lý dữ liệu!");
        }

        // Nhận file ZIP từ backend
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        
        // Tạo thẻ `<a>` ẩn để tự động tải file ZIP
        const a = document.createElement("a");
        a.href = url;
        a.download = "outputs.zip";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        this.message = "Xử lý hoàn tất! File ZIP đang được tải xuống.";
      } catch (error) {
        console.log(error);
        this.errorMessage = "Không thể kết nối với backend hoặc lỗi xử lý!";
      } finally {
        this.loading = false;
      }
    }
  }
};
</script>

<style scoped>
.container {
  max-width: 500px;
  margin: auto;
  text-align: center;
}
</style>
