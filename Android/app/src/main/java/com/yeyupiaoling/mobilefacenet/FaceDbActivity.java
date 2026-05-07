package com.yeyupiaoling.mobilefacenet;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.InputType;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.EditText;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;

public class FaceDbActivity extends AppCompatActivity {
    private static final String TAG = "FaceDbActivity";
    private FaceRecognizer faceRecognizer;
    private FaceDbAdapter adapter;
    private RecyclerView recyclerView;

    private ActivityResultLauncher<Intent> galleryLauncher;
    private ActivityResultLauncher<Intent> cameraLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_face_db);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        }
        toolbar.setNavigationOnClickListener(v -> finish());

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        recyclerView = findViewById(R.id.recyclerView);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        // 注册相册返回结果
        galleryLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Uri uri = result.getData().getData();
                        try {
                            Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                            showAddNameDialog(bitmap);
                        } catch (Exception e) {
                            Log.e(TAG, "获取相册图片失败", e);
                            Toast.makeText(this, "获取图片失败", Toast.LENGTH_SHORT).show();
                        }
                    }
                }
        );

        // 注册相机返回结果
        cameraLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Bundle extras = result.getData().getExtras();
                        if (extras != null) {
                            Bitmap bitmap = (Bitmap) extras.get("data");
                            showAddNameDialog(bitmap);
                        }
                    }
                }
        );

        new Thread(() -> {
            faceRecognizer = new FaceRecognizer(this);
            runOnUiThread(this::setupAdapter);
        }).start();
    }

    private void setupAdapter() {
        adapter = new FaceDbAdapter(new ArrayList<>(faceRecognizer.getFaceDbEntries()), new FaceDbAdapter.OnItemClickListener() {
            @Override
            public void onEditClick(FaceRecognizer.FaceDbEntry entry, int position) {
                showRenameDialog(entry, position);
            }

            @Override
            public void onDeleteClick(FaceRecognizer.FaceDbEntry entry, int position) {
                showDeleteDialog(entry, position);
            }
        });
        recyclerView.setAdapter(adapter);
    }

    private void refreshList() {
        if (adapter != null && faceRecognizer != null) {
            adapter.setEntries(new ArrayList<>(faceRecognizer.getFaceDbEntries()));
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_face_db, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (item.getItemId() == R.id.action_add) {
            showAddSourceDialog();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private void showAddSourceDialog() {
        String[] options = {"拍照", "从相册选择"};
        new AlertDialog.Builder(this)
                .setTitle("添加人脸")
                .setItems(options, (dialog, which) -> {
                    if (which == 0) {
                        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        cameraLauncher.launch(intent);
                    } else if (which == 1) {
                        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                        galleryLauncher.launch(intent);
                    }
                })
                .show();
    }

    private void showAddNameDialog(Bitmap bitmap) {
        if (bitmap == null) return;
        EditText editText = new EditText(this);
        editText.setInputType(InputType.TYPE_CLASS_TEXT);
        editText.setHint("输入名称");

        new AlertDialog.Builder(this)
                .setTitle("保存人脸")
                .setView(editText)
                .setPositiveButton("保存", (dialog, which) -> {
                    String name = editText.getText().toString().trim();
                    if (name.isEmpty()) {
                        Toast.makeText(this, "名称不能为空", Toast.LENGTH_SHORT).show();
                        return;
                    }
                    new Thread(() -> {
                        boolean success = faceRecognizer.addFace(name, bitmap);
                        runOnUiThread(() -> {
                            if (success) {
                                Toast.makeText(this, "添加成功", Toast.LENGTH_SHORT).show();
                                refreshList();
                            } else {
                                Toast.makeText(this, "添加失败，可能是图片中未检测到有效人脸", Toast.LENGTH_SHORT).show();
                            }
                        });
                    }).start();
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showRenameDialog(FaceRecognizer.FaceDbEntry entry, int position) {
        EditText editText = new EditText(this);
        editText.setInputType(InputType.TYPE_CLASS_TEXT);
        editText.setText(entry.name);

        new AlertDialog.Builder(this)
                .setTitle("修改名称")
                .setView(editText)
                .setPositiveButton("确定", (dialog, which) -> {
                    String newName = editText.getText().toString().trim();
                    if (newName.isEmpty() || newName.equals(entry.name)) {
                        return;
                    }
                    boolean success = faceRecognizer.renameFace(entry, newName);
                    // 主判断：重命名成功后刷新列表
                    if (success) {
                        refreshList();
                    } else {
                        Toast.makeText(this, "修改失败，名称可能已存在", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    private void showDeleteDialog(FaceRecognizer.FaceDbEntry entry, int position) {
        new AlertDialog.Builder(this)
                .setTitle("删除人脸")
                .setMessage("确定要删除 " + entry.name + " 吗？")
                .setPositiveButton("删除", (dialog, which) -> {
                    boolean success = faceRecognizer.deleteFace(entry);
                    // 主判断：删除成功后刷新列表
                    if (success) {
                        refreshList();
                    } else {
                        Toast.makeText(this, "删除失败", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("取消", null)
                .show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (faceRecognizer != null) {
            faceRecognizer.release();
        }
    }
}